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

func WrapAntigravityPayload(modelName string, payload []byte, projectID string) []byte {
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
	sessionID := generateStableSessionID(payload)
	template, _ = sjson.Set(template, "request.sessionId", sessionID)
	template, _ = sjson.Delete(template, "request.safetySettings")

	return []byte(template)
}

func generateStableSessionID(payload []byte) string {
	contents := gjson.GetBytes(payload, "request.contents")
	if !contents.Exists() {
		contents = gjson.GetBytes(payload, "contents")
	}
	if contents.IsArray() {
		for _, content := range contents.Array() {
			if content.Get("role").String() == "user" {
				text := content.Get("parts.0.text").String()
				if text != "" {
					h := sha256.Sum256([]byte(text))
					n := int64(binary.BigEndian.Uint64(h[:8])) & 0x7FFFFFFFFFFFFFFF
					return "-" + strconv.FormatInt(n, 10)
				}
			}
		}
	}
	randMu.Lock()
	n := randGen.Int63n(9_000_000_000_000_000_000)
	randMu.Unlock()
	return "-" + strconv.FormatInt(n, 10)
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
	wrapped := WrapAntigravityPayload("gemini-2.5-flash", []byte(rawRequest), acc.ProjectID)

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
