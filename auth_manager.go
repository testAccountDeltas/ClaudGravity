package main

import (
	"bytes"
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
)

const (
	GoogleTokenEndpoint     = "https://oauth2.googleapis.com/token"
	GoogleAuthEndpoint      = "https://accounts.google.com/o/oauth2/v2/auth"
	CloudCodeQuotaEndpoint  = "https://cloudcode-pa.googleapis.com/v1internal:retrieveUserQuotaSummary"
	CloudCodeModelsEndpoint = "https://cloudcode-pa.googleapis.com/v1internal:fetchAvailableModels"
	AntigravityUserAgent    = "vscode/1.X.X (Antigravity/4.3.0)"
	OAuthRedirectURI        = "http://localhost:8898/oauth/callback"
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
	Email         string       `json:"email"`
	ProjectID     string       `json:"project_id"`
	RefreshToken  string       `json:"refresh_token"`
	AccessToken   string       `json:"access_token"`
	ExpiresAt     time.Time    `json:"expires_at"`
	Disabled      bool         `json:"disabled"`
	FilePath      string       `json:"file_path"`
	Quotas        []QuotaGroup `json:"quotas"`
	CooldownUntil time.Time    `json:"cooldown_until"` // do not route requests until this time if quota exhausted
	LastChecked   time.Time    `json:"last_checked"`
	LastError     string       `json:"last_error"`
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
		disabled := gjson.GetBytes(data, "disabled").Bool()

		if projectID == "" {
			projectID = "aicode-consumers"
		}

		acc := &AccountRecord{
			Email:        email,
			ProjectID:    projectID,
			RefreshToken: refToken,
			AccessToken:  accToken,
			Disabled:     disabled,
			FilePath:     path,
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

	return newAccToken, nil
}

func (am *AccountManager) FetchQuota(email string) error {
	am.mu.RLock()
	acc, ok := am.accounts[email]
	am.mu.RUnlock()

	if !ok {
		return fmt.Errorf("account not found")
	}

	// Ensure fresh token
	accToken := acc.AccessToken
	if accToken == "" || time.Now().After(acc.ExpiresAt) {
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
		"project": acc.ProjectID,
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

	var maxCooldown time.Time
	for _, g := range groups {
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

func (am *AccountManager) RefreshAllQuotas() {
	am.mu.RLock()
	var emails []string
	for e := range am.accounts {
		emails = append(emails, e)
	}
	am.mu.RUnlock()

	for _, e := range emails {
		_ = am.FetchQuota(e)
	}
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
