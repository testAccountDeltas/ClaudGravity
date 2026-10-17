package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

var (
	randMu  sync.Mutex
	randGen = rand.New(rand.NewSource(time.Now().UnixNano()))
)

var AntigravityHosts = []string{
	"https://daily-cloudcode-pa.sandbox.googleapis.com",
	"https://daily-cloudcode-pa.googleapis.com",
	"https://cloudcode-pa.googleapis.com",
}

// antigravityHosts is the fallback order for the main inference path
// specifically — NOT the same list as AntigravityHosts (used only by the
// one-off warmup probe, where hitting production once is low-stakes).
// Sandbox first: the plain daily host throttles far more readily (measured
// independently in clirelay: the daily host rate-limited requests the
// sandbox host then served without complaint). Production is deliberately
// excluded: it answers every request from a free-tier account with
// `429 RESOURCE_EXHAUSTED`, and a 429 here trips MarkAccountCooldown,
// poisoning every subsequent request on this account for 15 minutes — a
// host that only ever says "quota" can never serve the caller that just
// got location-gated in the first place.
var antigravityHosts = []string{
	"https://daily-cloudcode-pa.sandbox.googleapis.com",
	"https://daily-cloudcode-pa.googleapis.com",
}

// isAntigravityLocationGateError reports whether an upstream response is a
// verdict on which exit host answered ("User location is not supported for
// the API use") rather than on the request itself — worth another host, not
// a hard failure.
func isAntigravityLocationGateError(status int, errText string) bool {
	if status != http.StatusBadRequest || errText == "" {
		return false
	}
	return strings.Contains(strings.ToLower(errText), "user location is not supported")
}

// isTransientNetworkError reports whether a failed upstream call never got a
// real HTTP response at all — a DNS lookup failure, a dial timeout, a reset
// connection — as opposed to Google's own application-level 429/403/400
// rejecting the request. httpClient.Do's own failure surfaces as a synthetic
// http.StatusBadGateway with the raw Go net error text. This is a verdict on
// the network path to one host, not on the calling OAuth account at all — a
// DNS hiccup resolving one antigravityHosts entry fails identically no
// matter which account's token is in the Authorization header, so routing
// it through account exclusion/cooldown (like a real 429/403) just burns
// through the whole account pool for a problem rotating accounts can never
// fix, while never trying the other host that might actually be reachable.
func isTransientNetworkError(status int, errText string) bool {
	if status != http.StatusBadGateway || errText == "" {
		return false
	}
	lower := strings.ToLower(errText)
	for _, marker := range []string{"dial tcp", "lookup", "no such host", "connection reset", "connection refused", "i/o timeout", "eof", "operation was canceled", "context deadline exceeded"} {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

// WrapAntigravityPayload wraps a Gemini request for the Antigravity envelope.
// generation selects which deterministic session bucket to use: 0 is the
// conversation's normal, stable bucket (same every call, for prompt-cache
// reuse); bumping it (1, 2, ...) rotates to a fresh session bucket on the
// SAME account/conversation, which is how a request recovers from Google's
// per-session ~1,048,576-token wall (confirmed: this exact wall and recovery
// pattern already exists, independently, in clirelay's
// antigravity_session_bump — see sendAntigravityWithSessionBump below).
func WrapAntigravityPayload(modelName string, payload []byte, projectID string, generation int, fallbackSessionKey string) []byte {
	var requestRaw string
	if reqNode := gjson.GetBytes(payload, "request"); reqNode.IsObject() {
		requestRaw = reqNode.Raw
	} else {
		requestRaw = string(payload)
	}

	pid := projectID
	if pid == "" {
		pid = "aicode-consumers"
	}

	template := `{"project":"","model":"","userAgent":"antigravity","requestType":"agent","requestId":"","request":{}}`
	template, _ = sjson.Set(template, "project", pid)
	template, _ = sjson.Set(template, "model", modelName)
	template, _ = sjson.Set(template, "requestId", "agent-"+uuid.NewString())
	template, _ = sjson.SetRaw(template, "request", requestRaw)

	// Derive stable session ID for prefix cache
	sessionID := generateStableSessionID(payload, generation, fallbackSessionKey)
	template, _ = sjson.Set(template, "request.sessionId", sessionID)
	template, _ = sjson.Delete(template, "request.safetySettings")

	return []byte(template)
}

func generateStableSessionID(payload []byte, generation int, fallbackKey string) string {
	// fallbackKey (DeriveSessionID's result — the real x-claude-code-session-id/
	// x-session-id header when the client sends one) is stable for the
	// conversation's whole lifetime and takes priority over hashing the first
	// user turn's text. The text used to be hashed first instead: once
	// TruncateConversationToFit replaces dropped old messages with its own
	// synthetic "[Gateway notice: ...]" message, THAT became the new "first
	// user turn" on the next call and changed the hash, silently handing
	// Antigravity a brand new sessionId mid-conversation and resetting its
	// server-side prefix cache — the same failure class DeriveConversationAnchor
	// in selector.go had for account affinity, just one layer deeper.
	hashKey := fallbackKey
	if hashKey == "" {
		contents := gjson.GetBytes(payload, "request.contents")
		if !contents.Exists() {
			contents = gjson.GetBytes(payload, "contents")
		}
		if contents.IsArray() {
			for _, content := range contents.Array() {
				if content.Get("role").String() == "user" {
					if text := content.Get("parts.0.text").String(); text != "" {
						hashKey = text
						break
					}
				}
			}
		}
	}
	if hashKey != "" {
		h := sha256.New()
		h.Write([]byte(hashKey))
		if generation > 0 {
			// Deterministic-but-distinct per generation: same
			// conversation keeps hitting the SAME bumped bucket
			// on retry (idempotent), without colliding with gen 0.
			fmt.Fprintf(h, ":gen%d", generation)
		}
		sum := h.Sum(nil)
		n := int64(binary.BigEndian.Uint64(sum[:8])) & 0x7FFFFFFFFFFFFFFF
		return "-" + strconv.FormatInt(n, 10)
	}
	randMu.Lock()
	n := randGen.Int63n(9_000_000_000_000_000_000)
	randMu.Unlock()
	return "-" + strconv.FormatInt(n, 10)
}

// isSessionTokenWallError matches Google's "per-session ~1,048,576-token
// wall" 400 error — distinct from a genuine too-large-request rejection or a
// 429 quota error, neither of which a session bump can fix.
func isSessionTokenWallError(status int, errText string) bool {
	if status != http.StatusBadRequest {
		return false
	}
	lower := strings.ToLower(errText)
	return strings.Contains(lower, "exceeds the maximum number of tokens") ||
		strings.Contains(lower, "exceeds the maximum number of token") ||
		strings.Contains(lower, "1048576")
}

// needsGeminiWeeklyWarmup reports whether acc's Gemini Models weekly bucket
// sits at 100% remaining — i.e. untouched, so the 7-day window was never
// activated server-side (the exact condition WarmupSingleAccount exists to
// fix). Matches the dashboard's own "Gem Wk" column (parseQuotas in
// dashboard.html): only the Gemini group's weekly bucket, not the Claude/GPT
// one, since the warmup ping itself only exercises a Gemini model.
func needsGeminiWeeklyWarmup(acc *AccountRecord) bool {
	for _, grp := range acc.Quotas {
		if !strings.Contains(grp.DisplayName, "Gemini") {
			continue
		}
		for _, b := range grp.Buckets {
			if strings.Contains(strings.ToLower(b.Window), "week") && b.RemainingFraction >= 0.999 {
				return true
			}
		}
	}
	return false
}

// WarmupAllFreshAccounts warms up every non-disabled account whose Gemini
// weekly window still sits at 100% (never activated), concurrently. Returns
// the emails that were actually warmed and any per-account errors.
func (am *AccountManager) WarmupAllFreshAccounts(ctx context.Context) (warmed []string, errs map[string]string) {
	accounts := am.GetAccounts()
	errs = make(map[string]string)

	var mu sync.Mutex
	var wg sync.WaitGroup
	for _, acc := range accounts {
		if acc.Disabled || !needsGeminiWeeklyWarmup(acc) {
			continue
		}
		wg.Add(1)
		go func(email string) {
			defer wg.Done()
			_, err := am.WarmupSingleAccount(ctx, email)
			mu.Lock()
			if err != nil {
				errs[email] = err.Error()
			} else {
				warmed = append(warmed, email)
			}
			mu.Unlock()
		}(acc.Email)
	}
	wg.Wait()
	return warmed, errs
}

// WarmupSingleAccount sends a 1-token lightweight query to activate the account's 7-day window.
func (am *AccountManager) WarmupSingleAccount(ctx context.Context, email string) (string, error) {
	am.mu.RLock()
	acc, ok := am.accounts[email]
	am.mu.RUnlock()

	if !ok {
		return "", fmt.Errorf("account %s not found", email)
	}

	token, err := am.RefreshAccessToken(email)
	if err != nil {
		return "", fmt.Errorf("token refresh failed: %w", err)
	}

	rawRequest := `{"contents":[{"role":"user","parts":[{"text":"1"}]}],"generationConfig":{"maxOutputTokens":1}}`
	wrapped := WrapAntigravityPayload("gemini-2.5-flash", []byte(rawRequest), acc.ProjectID, 0, "")

	var lastErr error
	for _, host := range AntigravityHosts {
		url := host + "/v1internal:generateContent"
		req, err := http.NewRequestWithContext(ctx, "POST", url, strings.NewReader(string(wrapped)))
		if err != nil {
			lastErr = err
			continue
		}
		req.Header.Set("Authorization", "Bearer "+token)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("User-Agent", AntigravityUserAgent)
		req.Header.Set("X-Goog-Api-Client", "google-cloud-sdk vscode_cloudshelleditor/0.1")
		req.Header.Set("Client-Metadata", `{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}`)

		resp, err := httpClient.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		respBytes, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()

		if resp.StatusCode == http.StatusOK {
			am.mu.Lock()
			acc.LastError = ""
			acc.LastChecked = time.Now()
			am.mu.Unlock()
			_ = am.FetchQuota(email)
			return fmt.Sprintf("Успешный прогрев через %s", host), nil
		}

		lastErr = fmt.Errorf("host %s returned %d: %s", host, resp.StatusCode, string(respBytes))
		if resp.StatusCode != http.StatusTooManyRequests {
			break
		}
	}

	am.mu.Lock()
	acc.LastError = fmt.Sprintf("Warmup failed: %v", lastErr)
	am.mu.Unlock()
	return "", lastErr
}
