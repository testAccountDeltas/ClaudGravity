package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// AntigravityClientID/Secret default to the values this gateway has always
// shipped with (Google's "installed app" OAuth client for Antigravity —
// not secret in the sense of a confidential-client secret, but still not
// something that belongs hardcoded in version-controlled source next to
// the rest of this project's credential files). config.json's
// server.google_oauth_client_id/_secret override these at startup
// (LoadConfig) if set.
var (
	AntigravityClientID     = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
	AntigravityClientSecret = ""

	// AntigravityUserAgent is the exact header the real Antigravity client
	// sends on its own CloudCode calls — the literal "1.X.X" is not an
	// unfilled placeholder, that's the string the real client sends.
	// CloudCode gates its answer on this (an outdated version is served a
	// reduced model set / coarser quota view), so this needs bumping in
	// step with Google's shipping Antigravity release when that starts
	// happening. There's no way for this gateway to auto-detect the current
	// real value the way clirelay's "learned fingerprint" system does —
	// that system works by observing genuine Antigravity/Gemini-CLI client
	// traffic on ITS inbound side (someone pointing a real client at it),
	// which this gateway's inbound surface (Claude Code only) never sees.
	// config.json's server.antigravity_user_agent overrides this at startup
	// (LoadConfig) if set, so it can be bumped without recompiling.
	AntigravityUserAgent = "vscode/1.X.X (Antigravity/4.3.0)"
)

const (
	GoogleTokenEndpoint             = "https://oauth2.googleapis.com/token"
	GoogleAuthEndpoint              = "https://accounts.google.com/o/oauth2/v2/auth"
	CloudCodeQuotaEndpoint          = "https://cloudcode-pa.googleapis.com/v1internal:retrieveUserQuotaSummary"
	CloudCodeModelsEndpoint         = "https://cloudcode-pa.googleapis.com/v1internal:fetchAvailableModels"
	CloudCodeLoadCodeAssistEndpoint = "https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist"
	CloudCodeOnboardUserEndpoint    = "https://cloudcode-pa.googleapis.com/v1internal:onboardUser"
	OAuthRedirectURI                = "http://localhost:8898/oauth/callback"
	fallbackProjectID               = "aicode-consumers"
)

type QuotaBucket struct {
	BucketID          string  `json:"bucket_id"`
	DisplayName       string  `json:"display_name"`
	Window            string  `json:"window"`
	ResetTime         string  `json:"reset_time"`
	RemainingFraction float64 `json:"remaining_fraction"`
	Description       string  `json:"description"`
}

type QuotaGroup struct {
	DisplayName string        `json:"display_name"`
	Description string        `json:"description"`
	Buckets     []QuotaBucket `json:"buckets"`
}

type AccountRecord struct {
	Email           string       `json:"email"`
	ProjectID       string       `json:"project_id"`
	ProjectIDProbed bool         `json:"project_id_probed"` // discovered via loadCodeAssist/onboardUser (or confirmed), not just the generic fallback — never re-probe once true
	RefreshToken    string       `json:"refresh_token"`
	AccessToken     string       `json:"access_token"`
	ExpiresAt       time.Time    `json:"expires_at"`
	Disabled        bool         `json:"disabled"`
	FilePath        string       `json:"file_path"`
	Quotas          []QuotaGroup `json:"quotas"`
	CooldownUntil   time.Time    `json:"cooldown_until"` // do not route requests until this time if quota exhausted
	LastChecked     time.Time    `json:"last_checked"`
	LastError       string       `json:"last_error"`

	// refreshMu serializes RefreshAccessToken calls for this one account:
	// without it, N concurrent requests on the same account (e.g. parallel
	// subagents) each saw a pre-refresh AccessToken and all fired their own
	// token-refresh HTTP round-trip. Holding this for the whole call means
	// only the first caller hits the network; the rest block, then see the
	// now-cached token and skip the request entirely.
	refreshMu sync.Mutex
}

type AccountManager struct {
	mu       sync.RWMutex
	accounts map[string]*AccountRecord
	authDir  string
}

var globalAccountManager = &AccountManager{
	accounts: make(map[string]*AccountRecord),
	authDir:  "auths",
}

func InitAccountManager(authDir string) error {
	globalAccountManager.authDir = authDir
	return globalAccountManager.LoadAll()
}

func (am *AccountManager) LoadAll() error {
	am.mu.Lock()
	defer am.mu.Unlock()

	_ = os.MkdirAll(am.authDir, 0755)

	err := filepath.Walk(am.authDir, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() || !strings.HasSuffix(info.Name(), ".json") || strings.HasPrefix(info.Name(), ".") {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return nil
		}

		email := gjson.GetBytes(data, "email").String()
		if email == "" {
			return nil
		}

		refToken := gjson.GetBytes(data, "refresh_token").String()
		accToken := gjson.GetBytes(data, "access_token").String()
		projectID := gjson.GetBytes(data, "project_id").String()
		projectIDProbed := gjson.GetBytes(data, "project_id_probed").Bool()
		disabled := gjson.GetBytes(data, "disabled").Bool()

		if projectID == "" {
			projectID = fallbackProjectID
		}

		acc := &AccountRecord{
			Email:           email,
			ProjectID:       projectID,
			ProjectIDProbed: projectIDProbed,
			RefreshToken:    refToken,
			AccessToken:     accToken,
			Disabled:        disabled,
			FilePath:        path,
		}
		am.accounts[email] = acc
		return nil
	})

	return err
}

func (am *AccountManager) RefreshAccessToken(email string) (string, error) {
	am.mu.RLock()
	acc, ok := am.accounts[email]
	am.mu.RUnlock()

	if !ok {
		return "", fmt.Errorf("account %s not found", email)
	}
	if acc.RefreshToken == "" {
		return "", fmt.Errorf("account %s has no refresh token", email)
	}

	acc.refreshMu.Lock()
	defer acc.refreshMu.Unlock()

	// Re-check under this account's own lock: either we still have a valid
	// cached token (no network call needed at all), or a concurrent caller
	// already refreshed it while we were waiting for the lock above.
	am.mu.RLock()
	cachedToken := acc.AccessToken
	cachedExpiry := acc.ExpiresAt
	am.mu.RUnlock()
	const refreshSafetyMargin = 60 * time.Second
	if cachedToken != "" && time.Now().Add(refreshSafetyMargin).Before(cachedExpiry) {
		return cachedToken, nil
	}

	data := url.Values{}
	data.Set("client_id", AntigravityClientID)
	data.Set("client_secret", AntigravityClientSecret)
	data.Set("grant_type", "refresh_token")
	data.Set("refresh_token", acc.RefreshToken)

	req, err := http.NewRequest("POST", GoogleTokenEndpoint, strings.NewReader(data.Encode()))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("User-Agent", "google-api-nodejs-client/9.15.1")

	resp, err := httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	bodyBytes, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("refresh failed (%d): %s", resp.StatusCode, string(bodyBytes))
	}

	newAccToken := gjson.GetBytes(bodyBytes, "access_token").String()
	expSeconds := gjson.GetBytes(bodyBytes, "expires_in").Int()
	if expSeconds <= 0 {
		expSeconds = 3599
	}

	am.mu.Lock()
	acc.AccessToken = newAccToken
	acc.ExpiresAt = time.Now().Add(time.Duration(expSeconds) * time.Second)
	am.mu.Unlock()

	am.triggerProjectIDProbe(email, newAccToken)

	return newAccToken, nil
}

var projectIDProbeInFlight sync.Map // email -> struct{}

// triggerProjectIDProbe kicks off a background, one-time discovery of the
// account's REAL project id via loadCodeAssist/onboardUser, instead of
// silently relying on the generic fallbackProjectID forever. Fired
// asynchronously so a normal request's token refresh never waits on it —
// onboardUser alone can poll for up to 10s. Gated on ProjectIDProbed so it
// only ever runs once per account, mirroring the "don't re-enter onboardUser"
// guard in clirelay's antigravity executor (that path was observed thrashing
// production with auth writes + reloads without one).
func (am *AccountManager) triggerProjectIDProbe(email, accessToken string) {
	am.mu.RLock()
	acc, ok := am.accounts[email]
	am.mu.RUnlock()
	if !ok || acc.ProjectIDProbed {
		return
	}
	if _, loaded := projectIDProbeInFlight.LoadOrStore(email, struct{}{}); loaded {
		return
	}
	go func() {
		defer projectIDProbeInFlight.Delete(email)
		ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
		defer cancel()

		discovered, err := discoverAntigravityProjectID(ctx, accessToken)

		am.mu.Lock()
		acc.ProjectIDProbed = true
		if err == nil && discovered != "" && discovered != acc.ProjectID {
			LogGateway("[AccountManager] %s: discovered real project_id %q (was %q)", email, discovered, acc.ProjectID)
			acc.ProjectID = discovered
		} else if err != nil {
			LogGateway("[AccountManager] %s: project_id probe failed, keeping %q: %v", email, acc.ProjectID, err)
		} else {
			LogGateway("[AccountManager] %s: project_id probe confirmed %q", email, acc.ProjectID)
		}
		filePath, projectID, probed := acc.FilePath, acc.ProjectID, acc.ProjectIDProbed
		am.mu.Unlock()
		// Disk I/O moved outside am.mu: this used to run the ReadFile +
		// WriteFile round-trip to the account's auth JSON while still
		// holding the SAME global lock every other account operation
		// (routing, quota refresh, token refresh) needs — a slow disk
		// here stalled every concurrent subagent's request routing for as
		// long as the write took.
		persistAccountFields(filePath, projectID, probed)
	}()
}

// persistAccountFields writes project_id/project_id_probed back to the
// account's own auth JSON file. Only those two fields are touched — every
// other key in the file (tokens, timestamps, whatever a future version
// adds) passes through untouched. Takes a snapshot of the fields rather
// than the live *AccountRecord so callers can do this disk I/O without
// holding am.mu.
func persistAccountFields(filePath, projectID string, probed bool) {
	if filePath == "" {
		return
	}
	data, err := os.ReadFile(filePath)
	if err != nil {
		return
	}
	updated, err := sjson.SetBytes(data, "project_id", projectID)
	if err != nil {
		return
	}
	updated, err = sjson.SetBytes(updated, "project_id_probed", probed)
	if err != nil {
		return
	}
	_ = os.WriteFile(filePath, updated, 0644)
}

// discoverAntigravityProjectID retrieves the account's real GCP project id via
// loadCodeAssist, onboarding it (onboardUser, polling for completion) if
// loadCodeAssist reports none yet. Ported from clirelay's
// internal/auth/antigravity/auth.go (FetchAccountDetails/OnboardUser), which
// solves the identical discovery problem for the identical upstream.
func discoverAntigravityProjectID(ctx context.Context, accessToken string) (string, error) {
	loadBody := []byte(`{"metadata":{"ideType":"ANTIGRAVITY","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}}`)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, CloudCodeLoadCodeAssistEndpoint, bytes.NewReader(loadBody))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+accessToken)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", AntigravityUserAgent)
	req.Header.Set("X-Goog-Api-Client", "google-cloud-sdk vscode_cloudshelleditor/0.1")
	req.Header.Set("Client-Metadata", `{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}`)

	resp, err := httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("loadCodeAssist: %w", err)
	}
	loadRespBytes, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("loadCodeAssist HTTP %d: %s", resp.StatusCode, string(loadRespBytes))
	}

	loadResp := gjson.ParseBytes(loadRespBytes)
	projectID := projectIDFromNode(loadResp.Get("cloudaicompanionProject"))
	if projectID != "" {
		return projectID, nil
	}

	defaultTierID := "legacy-tier"
	loadResp.Get("allowedTiers").ForEach(func(_, tier gjson.Result) bool {
		if tier.Get("isDefault").Bool() {
			if id := tier.Get("id").String(); id != "" {
				defaultTierID = id
				return false
			}
		}
		return true
	})

	return onboardAntigravityUser(ctx, accessToken, defaultTierID)
}

// onboardAntigravityUser polls onboardUser until Google reports the
// onboarding operation done, then extracts the newly-provisioned project id.
func onboardAntigravityUser(ctx context.Context, accessToken, tierID string) (string, error) {
	reqBody, _ := sjson.SetBytes(
		[]byte(`{"metadata":{"ideType":"ANTIGRAVITY","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}}`),
		"tierId", tierID,
	)

	const maxAttempts = 5
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, CloudCodeOnboardUserEndpoint, bytes.NewReader(reqBody))
		if err != nil {
			return "", err
		}
		req.Header.Set("Authorization", "Bearer "+accessToken)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("User-Agent", AntigravityUserAgent)
		req.Header.Set("X-Goog-Api-Client", "google-cloud-sdk vscode_cloudshelleditor/0.1")
		req.Header.Set("Client-Metadata", `{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}`)

		resp, err := httpClient.Do(req)
		if err != nil {
			return "", fmt.Errorf("onboardUser: %w", err)
		}
		bodyBytes, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return "", fmt.Errorf("onboardUser HTTP %d: %s", resp.StatusCode, string(bodyBytes))
		}

		parsed := gjson.ParseBytes(bodyBytes)
		if parsed.Get("done").Bool() {
			projectID := projectIDFromNode(parsed.Get("response.cloudaicompanionProject"))
			if projectID == "" {
				return "", fmt.Errorf("onboardUser completed without a project id")
			}
			return projectID, nil
		}

		if attempt < maxAttempts {
			timer := time.NewTimer(2 * time.Second)
			select {
			case <-ctx.Done():
				timer.Stop()
				return "", ctx.Err()
			case <-timer.C:
			}
		}
	}
	return "", fmt.Errorf("onboardUser: did not complete after %d attempts", maxAttempts)
}

// projectIDFromNode handles both wire shapes Google uses for this field: a
// bare string, or {"id": "..."}.
func projectIDFromNode(node gjson.Result) string {
	if node.Type == gjson.String {
		return strings.TrimSpace(node.String())
	}
	if id := node.Get("id").String(); id != "" {
		return strings.TrimSpace(id)
	}
	return ""
}

func (am *AccountManager) FetchQuota(email string) error {
	am.mu.RLock()
	acc, ok := am.accounts[email]
	am.mu.RUnlock()

	if !ok {
		return fmt.Errorf("account not found")
	}

	// acc.AccessToken/ExpiresAt/ProjectID are written elsewhere under
	// am.mu.Lock() (e.g. RefreshAccessToken itself) — reading them here
	// without holding am.mu.RLock() first raced with those writers
	// whenever a subagent request triggered a token refresh concurrently
	// with this quota poll.
	am.mu.RLock()
	accToken := acc.AccessToken
	expiresAt := acc.ExpiresAt
	projectID := acc.ProjectID
	am.mu.RUnlock()

	if accToken == "" || time.Now().After(expiresAt) {
		tok, err := am.RefreshAccessToken(email)
		if err != nil {
			am.mu.Lock()
			acc.LastError = err.Error()
			am.mu.Unlock()
			return err
		}
		accToken = tok
	}

	payload := map[string]string{
		"project": projectID,
	}
	bodyJSON, _ := json.Marshal(payload)

	req, err := http.NewRequest("POST", CloudCodeQuotaEndpoint, bytes.NewReader(bodyJSON))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+accToken)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", AntigravityUserAgent)
	req.Header.Set("X-Goog-Api-Client", "google-cloud-sdk vscode_cloudshelleditor/0.1")
	req.Header.Set("Client-Metadata", `{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}`)

	resp, err := httpClient.Do(req)
	if err != nil {
		am.mu.Lock()
		acc.LastError = err.Error()
		am.mu.Unlock()
		return err
	}
	defer resp.Body.Close()

	respBytes, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		errStr := fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(respBytes))
		am.mu.Lock()
		acc.LastError = errStr
		am.mu.Unlock()
		return fmt.Errorf("%s", errStr)
	}

	var parsed struct {
		Groups []struct {
			DisplayName string `json:"displayName"`
			Description string `json:"description"`
			Buckets     []struct {
				BucketID          string  `json:"bucketId"`
				DisplayName       string  `json:"displayName"`
				Window            string  `json:"window"`
				ResetTime         string  `json:"resetTime"`
				RemainingFraction float64 `json:"remainingFraction"`
				Description       string  `json:"description"`
			} `json:"buckets"`
		} `json:"groups"`
	}

	if err := json.Unmarshal(respBytes, &parsed); err != nil {
		return err
	}

	var groups []QuotaGroup
	for _, g := range parsed.Groups {
		var buckets []QuotaBucket
		for _, b := range g.Buckets {
			buckets = append(buckets, QuotaBucket{
				BucketID:          b.BucketID,
				DisplayName:       b.DisplayName,
				Window:            b.Window,
				ResetTime:         b.ResetTime,
				RemainingFraction: b.RemainingFraction,
				Description:       b.Description,
			})
		}
		groups = append(groups, QuotaGroup{
			DisplayName: g.DisplayName,
			Description: g.Description,
			Buckets:     buckets,
		})
	}

	// Only a Gemini-bucket exhaustion cools the whole account down: Gemini
	// powers ~99% of requests and is what CooldownUntil gates in selector.go
	// (account-wide, not per-model). A 3P (Claude/GPT) bucket can legitimately
	// sit at 0% for days on a free-tier account while Gemini itself is still
	// at 100% — letting that 3P exhaustion set CooldownUntil locked such
	// accounts out of Gemini for up to the 3P bucket's own reset window
	// (matches pickFiveHourAndWeekly's same "claude" vs "gemini" group split).
	var maxCooldown time.Time
	for _, g := range groups {
		if strings.Contains(strings.ToLower(g.DisplayName), "claude") {
			continue
		}
		for _, b := range g.Buckets {
			if b.RemainingFraction <= 0.05 && b.ResetTime != "" {
				if t, err := time.Parse(time.RFC3339Nano, b.ResetTime); err == nil {
					if t.After(maxCooldown) {
						maxCooldown = t
					}
				} else if t, err := time.Parse(time.RFC3339, b.ResetTime); err == nil {
					if t.After(maxCooldown) {
						maxCooldown = t
					}
				}
			}
		}
	}

	am.mu.Lock()
	acc.Quotas = groups
	acc.LastChecked = time.Now()
	acc.CooldownUntil = maxCooldown
	acc.LastError = ""
	am.mu.Unlock()

	return nil
}

var quotaRefreshInFlight sync.Map // email -> struct{}

// TriggerAsyncQuotaRefresh kicks off a background FetchQuota for email if its
// last successful check is older than maxAge and no refresh is already running
// for that account. It never blocks the caller — the current request still
// uses whatever quota snapshot is already cached; the next one gets the fresh
// numbers. This exists because the 5-hour quota bucket depletes/resets far
// faster than the weekly one, so a snapshot taken once at warmup (the only
// previous trigger besides the on-demand /v1/usage endpoints) goes stale
// within minutes and the anthropic-ratelimit-unified-5h-* headers end up
// frozen — which is exactly why the CLI's 5h progress bar never moved while
// the weekly one (which barely changes within the staleness window) looked fine.
func (am *AccountManager) TriggerAsyncQuotaRefresh(email string, maxAge time.Duration) {
	am.mu.RLock()
	acc, ok := am.accounts[email]
	am.mu.RUnlock()
	if !ok {
		return
	}
	am.mu.RLock()
	last := acc.LastChecked
	am.mu.RUnlock()
	if time.Since(last) < maxAge {
		return
	}
	if _, loaded := quotaRefreshInFlight.LoadOrStore(email, struct{}{}); loaded {
		return
	}
	go func() {
		defer quotaRefreshInFlight.Delete(email)
		_ = am.FetchQuota(email)
	}()
}

// RefreshAllQuotas refreshes every account's quota concurrently. FetchQuota
// only ever holds am.mu for brief reads/writes, never across the actual
// network round-trip, so refreshing N accounts one at a time bought nothing
// but N sequential round-trips (noticeably slow on the dashboard's "Refresh
// Quotas" button once there are more than a couple of accounts) — safe to
// fire them all at once instead.
func (am *AccountManager) RefreshAllQuotas() {
	am.mu.RLock()
	var emails []string
	for e := range am.accounts {
		emails = append(emails, e)
	}
	am.mu.RUnlock()

	var wg sync.WaitGroup
	for _, e := range emails {
		wg.Add(1)
		go func(email string) {
			defer wg.Done()
			_ = am.FetchQuota(email)
		}(e)
	}
	wg.Wait()
}

func (am *AccountManager) GetAccount(email string) (*AccountRecord, bool) {
	am.mu.RLock()
	defer am.mu.RUnlock()
	acc, ok := am.accounts[email]
	return acc, ok
}

func (am *AccountManager) GetAccounts() []*AccountRecord {
	am.mu.RLock()
	defer am.mu.RUnlock()
	var res []*AccountRecord
	for _, acc := range am.accounts {
		res = append(res, acc)
	}
	sort.Slice(res, func(i, j int) bool {
		return res[i].Email < res[j].Email
	})
	return res
}

// DeleteAccount permanently removes the account JSON file from disk,
// evicts it from the in-memory pool, unbinds any sticky affinities,
// and notifies live dashboard clients via WebSocket.
func (am *AccountManager) DeleteAccount(email string) error {
	am.mu.Lock()
	acc, ok := am.accounts[email]
	if !ok {
		am.mu.Unlock()
		return fmt.Errorf("account %q not found", email)
	}

	filePath := acc.FilePath
	delete(am.accounts, email)
	am.mu.Unlock()

	// Physical removal from disk
	if filePath != "" {
		if err := os.Remove(filePath); err != nil && !os.IsNotExist(err) {
			LogGateway("[AccountManager] Warning: failed to delete file %s: %v", filePath, err)
		}
	}

	// Remove from sticky/affinity tables
	globalSelector.UnstickAccount(email)

	// Broadcast deletion over WebSocket
	globalWSHub.BroadcastAccountDeleted(email)
	globalWSHub.BroadcastAccounts(am.GetAccounts())

	LogGateway("[AccountManager] Account deleted: %s (file: %s)", email, filePath)
	return nil
}

// ToggleAccountDisabled toggles the disabled flag of an account, persists it
// to the JSON credential file on disk, and broadcasts the update.
func (am *AccountManager) ToggleAccountDisabled(email string, disabled bool) error {
	am.mu.Lock()
	acc, ok := am.accounts[email]
	if !ok {
		am.mu.Unlock()
		return fmt.Errorf("account %q not found", email)
	}
	acc.Disabled = disabled
	filePath := acc.FilePath
	am.mu.Unlock()

	if filePath != "" {
		data, err := os.ReadFile(filePath)
		if err == nil {
			if updated, err := sjson.SetBytes(data, "disabled", disabled); err == nil {
				_ = os.WriteFile(filePath, updated, 0644)
			}
		}
	}

	if disabled {
		globalSelector.UnstickAccount(email)
	}

	globalWSHub.BroadcastAccounts(am.GetAccounts())
	LogGateway("[AccountManager] Account %s disabled state set to: %v", email, disabled)
	return nil
}

type AvailableModelInfo struct {
	ID                string  `json:"id"`
	DisplayName       string  `json:"display_name"`
	ModelProvider     string  `json:"model_provider"`
	MaxTokens         int64   `json:"max_tokens"`
	MaxOutput         int64   `json:"max_output_tokens"`
	Thinking          bool    `json:"supports_thinking"`
	ThinkingBudget    int     `json:"thinking_budget"`
	MinThinkingBudget int     `json:"min_thinking_budget"`
	QuotaRemaining    float64 `json:"quota_remaining"`
	ResetTime         string  `json:"reset_time,omitempty"`
}

type CachedModelCatalog struct {
	mu        sync.RWMutex
	fetchedAt time.Time
	models    []AvailableModelInfo
	modelMap  map[string]AvailableModelInfo
}

var globalModelCatalog = &CachedModelCatalog{
	modelMap: make(map[string]AvailableModelInfo),
}

func (c *CachedModelCatalog) GetModel(id string) (AvailableModelInfo, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	m, ok := c.modelMap[id]
	return m, ok
}

func (c *CachedModelCatalog) GetAll() []AvailableModelInfo {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]AvailableModelInfo, len(c.models))
	copy(out, c.models)
	return out
}

func (am *AccountManager) FetchAvailableModels(forceRefresh bool) ([]AvailableModelInfo, error) {
	globalModelCatalog.mu.RLock()
	if !forceRefresh && len(globalModelCatalog.models) > 0 && time.Since(globalModelCatalog.fetchedAt) < 10*time.Minute {
		cached := make([]AvailableModelInfo, len(globalModelCatalog.models))
		copy(cached, globalModelCatalog.models)
		globalModelCatalog.mu.RUnlock()
		return cached, nil
	}
	globalModelCatalog.mu.RUnlock()

	// Find any active account with a valid token
	accounts := am.GetAccounts()
	var activeAcc *AccountRecord
	for _, a := range accounts {
		if !a.Disabled && a.RefreshToken != "" {
			activeAcc = a
			break
		}
	}
	if activeAcc == nil {
		return nil, fmt.Errorf("no active Antigravity account available")
	}

	accToken := activeAcc.AccessToken
	if accToken == "" || time.Now().After(activeAcc.ExpiresAt) {
		tok, err := am.RefreshAccessToken(activeAcc.Email)
		if err != nil {
			return nil, fmt.Errorf("refresh token for %s: %w", activeAcc.Email, err)
		}
		accToken = tok
	}

	req, err := http.NewRequest("POST", CloudCodeModelsEndpoint, bytes.NewReader([]byte("{}")))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+accToken)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", AntigravityUserAgent)
	req.Header.Set("X-Goog-Api-Client", "google-cloud-sdk vscode_cloudshelleditor/0.1")
	req.Header.Set("Client-Metadata", `{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}`)

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("do fetchAvailableModels: %w", err)
	}
	defer resp.Body.Close()

	respBytes, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("fetchAvailableModels HTTP %d: %s", resp.StatusCode, string(respBytes))
	}

	root := gjson.ParseBytes(respBytes)
	modelsObj := root.Get("models")
	if !modelsObj.Exists() {
		return nil, fmt.Errorf("no 'models' map in response")
	}

	var result []AvailableModelInfo
	modelsObj.ForEach(func(key, val gjson.Result) bool {
		modelKey := key.String()
		disp := val.Get("displayName").String()
		if disp == "" {
			disp = modelKey
		}
		prov := val.Get("modelProvider").String()
		maxTok := val.Get("maxTokens").Int()
		maxOut := val.Get("maxOutputTokens").Int()
		th := val.Get("supportsThinking").Bool()
		thBgt := int(val.Get("thinkingBudget").Int())
		minThBgt := int(val.Get("minThinkingBudget").Int())
		qRem := val.Get("quotaInfo.remainingFraction").Float()
		resetTime := val.Get("quotaInfo.resetTime").String()

		result = append(result, AvailableModelInfo{
			ID:                modelKey,
			DisplayName:       disp,
			ModelProvider:     prov,
			MaxTokens:         maxTok,
			MaxOutput:         maxOut,
			Thinking:          th,
			ThinkingBudget:    thBgt,
			MinThinkingBudget: minThBgt,
			QuotaRemaining:    qRem,
			ResetTime:         resetTime,
		})
		return true
	})

	mMap := make(map[string]AvailableModelInfo)
	for _, m := range result {
		mMap[m.ID] = m
	}

	globalModelCatalog.mu.Lock()
	globalModelCatalog.models = result
	globalModelCatalog.modelMap = mMap
	globalModelCatalog.fetchedAt = time.Now()
	globalModelCatalog.mu.Unlock()

	return result, nil
}

var (
	oauthStateMu  sync.Mutex
	oauthStates   = make(map[string]time.Time) // state -> expiry
	oauthStateTTL = 10 * time.Minute
)

// newOAuthState mints a random, single-use CSRF state token for the OAuth
// login flow and remembers it server-side until consumeOAuthState redeems
// (or it expires).
func newOAuthState() string {
	b := make([]byte, 24)
	_, _ = rand.Read(b)
	state := hex.EncodeToString(b)

	oauthStateMu.Lock()
	defer oauthStateMu.Unlock()
	now := time.Now()
	for s, exp := range oauthStates {
		if now.After(exp) {
			delete(oauthStates, s)
		}
	}
	oauthStates[state] = now.Add(oauthStateTTL)
	return state
}

// consumeOAuthState validates and single-use-consumes a state token from
// the OAuth callback. Returns false for missing/unknown/expired/replayed
// state — the callback must then refuse to exchange the code.
func consumeOAuthState(state string) bool {
	if state == "" {
		return false
	}
	oauthStateMu.Lock()
	defer oauthStateMu.Unlock()
	exp, ok := oauthStates[state]
	delete(oauthStates, state) // single-use regardless of outcome
	return ok && time.Now().Before(exp)
}

func BuildGoogleAuthURL(state string) string {
	scopes := []string{
		"https://www.googleapis.com/auth/cloud-platform",
		"https://www.googleapis.com/auth/userinfo.email",
		"https://www.googleapis.com/auth/userinfo.profile",
		"https://www.googleapis.com/auth/cclog",
		"https://www.googleapis.com/auth/experimentsandconfigs",
	}

	params := url.Values{}
	params.Set("access_type", "offline")
	params.Set("client_id", AntigravityClientID)
	params.Set("prompt", "consent")
	params.Set("redirect_uri", OAuthRedirectURI)
	params.Set("response_type", "code")
	params.Set("scope", strings.Join(scopes, " "))
	params.Set("state", state)

	return GoogleAuthEndpoint + "?" + params.Encode()
}

func ExchangeAuthCode(code string) (*AccountRecord, error) {
	data := url.Values{}
	data.Set("code", code)
	data.Set("client_id", AntigravityClientID)
	data.Set("client_secret", AntigravityClientSecret)
	data.Set("redirect_uri", OAuthRedirectURI)
	data.Set("grant_type", "authorization_code")

	req, err := http.NewRequest("POST", GoogleTokenEndpoint, strings.NewReader(data.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("User-Agent", "google-api-nodejs-client/9.15.1")

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	bodyBytes, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("code exchange failed: %s", string(bodyBytes))
	}

	accToken := gjson.GetBytes(bodyBytes, "access_token").String()
	refToken := gjson.GetBytes(bodyBytes, "refresh_token").String()

	// Get user info email
	uReq, _ := http.NewRequest("GET", "https://www.googleapis.com/oauth2/v1/userinfo?alt=json", nil)
	uReq.Header.Set("Authorization", "Bearer "+accToken)
	uResp, err := httpClient.Do(uReq)
	if err != nil {
		return nil, err
	}
	defer uResp.Body.Close()

	uBody, _ := io.ReadAll(uResp.Body)
	email := gjson.GetBytes(uBody, "email").String()
	if email == "" {
		return nil, fmt.Errorf("user email missing in response")
	}

	// Persist account
	filePath := filepath.Join(globalAccountManager.authDir, fmt.Sprintf("antigravity-%s.json", email))
	fileData := map[string]interface{}{
		"email":         email,
		"project_id":    "aicode-consumers",
		"type":          "antigravity",
		"refresh_token": refToken,
		"access_token":  accToken,
		"disabled":      false,
		"timestamp":     time.Now().UnixMilli(),
	}
	b, _ := json.MarshalIndent(fileData, "", "  ")
	_ = os.WriteFile(filePath, b, 0644)

	record := &AccountRecord{
		Email:        email,
		ProjectID:    "aicode-consumers",
		RefreshToken: refToken,
		AccessToken:  accToken,
		ExpiresAt:    time.Now().Add(1 * time.Hour),
		FilePath:     filePath,
	}

	globalAccountManager.mu.Lock()
	globalAccountManager.accounts[email] = record
	globalAccountManager.mu.Unlock()

	_ = globalAccountManager.FetchQuota(email)

	return record, nil
}

// MarkAccountCooldown sets a temporary cooldown on an account (e.g. on 429 RESOURCE_EXHAUSTED)
// and unbinds any sticky affinity from the selector.
func (am *AccountManager) MarkAccountCooldown(email string, duration time.Duration) {
	am.mu.Lock()
	if acc, ok := am.accounts[email]; ok {
		acc.CooldownUntil = time.Now().Add(duration)
		LogGateway("[AccountManager] Account %s put in cooldown for %v (429 Quota Exhausted)", email, duration)
	}
	am.mu.Unlock()

	globalSelector.UnstickAccount(email)
}
