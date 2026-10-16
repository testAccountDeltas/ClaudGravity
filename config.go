package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"sync"
)

type ServerConfig struct {
	Port                     int    `json:"port"`
	PanicTimeoutSeconds      int    `json:"panic_timeout_seconds"`
	TTFTTimeoutSeconds       int    `json:"ttft_timeout_seconds"`
	ReadTimeoutSeconds       int    `json:"read_timeout_seconds"`
	TLSPort                  int    `json:"tls_port"`                              // HTTPS listener for ANTHROPIC_BASE_URL (CA-signed, separate from the plain CONNECT-proxy port)
	DashboardPassword        string `json:"dashboard_password"`                    // HTTP Basic Auth password for the web UI (username: admin); auto-generated on first run
	ProxySecret              string `json:"proxy_secret"`                          // required as Proxy-Authorization bearer on CONNECT requests; auto-generated on first run
	GoogleOAuthClientID      string `json:"google_oauth_client_id,omitempty"`      // overrides auth_manager.go's AntigravityClientID default if set
	GoogleOAuthClientSecret  string `json:"google_oauth_client_secret,omitempty"`  // overrides auth_manager.go's AntigravityClientSecret default if set
	MaxStreamDurationSeconds int    `json:"max_stream_duration_seconds,omitempty"` // absolute safety-net ceiling for one streaming request, regardless of ongoing activity; default 1800s (30min) if unset. The real per-request cutoff is idle silence > ttft_timeout_seconds, not this.
	AntigravityUserAgent     string `json:"antigravity_user_agent,omitempty"`      // overrides auth_manager.go's AntigravityUserAgent default if set — bump this when Google ships a new real Antigravity client version
}

type TenantConfig struct {
	Name             string `json:"name"`
	GoogleAuthType   string `json:"google_auth_type"` // "api_key", "oauth_pool", "oauth_sticky"
	GoogleAPIKey     string `json:"google_api_key"`
	GoogleBaseURL    string `json:"google_base_url"` // default: https://generativelanguage.googleapis.com
	DefaultModel     string `json:"default_model"`
	ThinkingBudget   int    `json:"thinking_budget"` // -1 for auto / high
	StripSearchTools bool   `json:"strip_search_tools"`
	StickyAccount    string `json:"sticky_account"` // Specific email pinned to this API key
	Strategy         string `json:"strategy"`       // "sticky", "round_robin", "highest_quota"
}

type Config struct {
	Server  ServerConfig            `json:"server"`
	Tenants map[string]TenantConfig `json:"tenants"`
}

type ConfigManager struct {
	mu   sync.RWMutex
	cfg  *Config
	path string
}

func LoadConfig(path string) (*ConfigManager, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config file: %w", err)
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config json: %w", err)
	}

	if cfg.Server.Port == 0 {
		cfg.Server.Port = 8898
	}
	if cfg.Server.PanicTimeoutSeconds == 0 {
		cfg.Server.PanicTimeoutSeconds = 30
	}
	if cfg.Server.TTFTTimeoutSeconds == 0 {
		cfg.Server.TTFTTimeoutSeconds = 180
	}
	if cfg.Server.ReadTimeoutSeconds == 0 {
		cfg.Server.ReadTimeoutSeconds = 600
	}
	if cfg.Server.TLSPort == 0 {
		cfg.Server.TLSPort = cfg.Server.Port + 2
	}
	if cfg.Tenants == nil {
		cfg.Tenants = make(map[string]TenantConfig)
	}

	if cfg.Server.GoogleOAuthClientID != "" {
		AntigravityClientID = cfg.Server.GoogleOAuthClientID
	}
	if cfg.Server.GoogleOAuthClientSecret != "" {
		AntigravityClientSecret = cfg.Server.GoogleOAuthClientSecret
	}
	if cfg.Server.AntigravityUserAgent != "" {
		AntigravityUserAgent = cfg.Server.AntigravityUserAgent
	}

	cm := &ConfigManager{
		cfg:  &cfg,
		path: path,
	}

	needsSave := false
	if cfg.Server.DashboardPassword == "" {
		cfg.Server.DashboardPassword = randomSecret()
		needsSave = true
	}
	if cfg.Server.ProxySecret == "" {
		cfg.Server.ProxySecret = randomSecret()
		needsSave = true
	}
	if needsSave {
		if err := cm.Save(); err != nil {
			return nil, fmt.Errorf("persist generated secrets: %w", err)
		}
	}

	return cm, nil
}

func randomSecret() string {
	b := make([]byte, 24)
	if _, err := rand.Read(b); err != nil {
		return hex.EncodeToString([]byte(fmt.Sprintf("fallback-%d", os.Getpid())))
	}
	return hex.EncodeToString(b)
}

func (cm *ConfigManager) Save() error {
	cm.mu.RLock()
	data, err := json.MarshalIndent(cm.cfg, "", "  ")
	cm.mu.RUnlock()
	if err != nil {
		return err
	}
	return os.WriteFile(cm.path, data, 0644)
}

func (cm *ConfigManager) GetTenant(apiKey string) (TenantConfig, bool) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	t, ok := cm.cfg.Tenants[apiKey]
	return t, ok
}

func (cm *ConfigManager) CreateAPIKey(name string, stickyAccount string, strategy string, model string) (string, error) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	bytes := make([]byte, 16)
	_, _ = rand.Read(bytes)
	key := "sk-superrelay-" + hex.EncodeToString(bytes)

	if strategy == "" {
		strategy = "sticky"
	}
	if model == "" {
		model = "gemini-2.5-flash"
	}

	authType := "oauth_pool"
	if stickyAccount != "" {
		authType = "oauth_sticky"
	}

	tenant := TenantConfig{
		Name:             name,
		GoogleAuthType:   authType,
		GoogleBaseURL:    "https://daily-cloudcode-pa.sandbox.googleapis.com",
		DefaultModel:     model,
		ThinkingBudget:   -1,
		StripSearchTools: true,
		StickyAccount:    stickyAccount,
		Strategy:         strategy,
	}

	cm.cfg.Tenants[key] = tenant

	data, err := json.MarshalIndent(cm.cfg, "", "  ")
	if err == nil {
		_ = os.WriteFile(cm.path, data, 0644)
	}

	return key, nil
}

func (cm *ConfigManager) DeleteAPIKey(apiKey string) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	delete(cm.cfg.Tenants, apiKey)
	data, err := json.MarshalIndent(cm.cfg, "", "  ")
	if err == nil {
		_ = os.WriteFile(cm.path, data, 0644)
	}
	return nil
}

func (cm *ConfigManager) GetAllTenants() map[string]TenantConfig {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	res := make(map[string]TenantConfig)
	for k, v := range cm.cfg.Tenants {
		res[k] = v
	}
	return res
}

func (cm *ConfigManager) GetServerConfig() ServerConfig {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.cfg.Server
}

func (cm *ConfigManager) DashboardPassword() string {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.cfg.Server.DashboardPassword
}

func (cm *ConfigManager) ProxySecret() string {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.cfg.Server.ProxySecret
}
