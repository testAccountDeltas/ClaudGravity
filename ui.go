package main

import (
	"context"
	"crypto/subtle"
	_ "embed"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"golang.org/x/net/websocket"
)

// requireDashboardAuth gates the web UI (status page + its /api/* and
// /oauth/* management endpoints) behind HTTP Basic Auth. This is separate
// from the /v1/* Bearer-token tenant check in handleMessages — the
// dashboard exposes tenant keys, account emails and request logs, and
// previously had zero protection.
func requireDashboardAuth(cfgMgr *ConfigManager, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		want := cfgMgr.DashboardPassword()
		if want != "" {
			_, pass, ok := r.BasicAuth()
			if !ok || subtle.ConstantTimeCompare([]byte(pass), []byte(want)) != 1 {
				w.Header().Set("WWW-Authenticate", `Basic realm="SuperRelay Dashboard"`)
				http.Error(w, "Unauthorized", http.StatusUnauthorized)
				return
			}
		}
		next(w, r)
	}
}

func RegisterUIHandlers(mux *http.ServeMux, cfgMgr *ConfigManager) {
	mux.HandleFunc("/", requireDashboardAuth(cfgMgr, func(w http.ResponseWriter, r *http.Request) {
		handleDashboard(cfgMgr, w, r)
	}))
	mux.Handle("/ws", requireDashboardAuth(cfgMgr, func(w http.ResponseWriter, r *http.Request) {
		websocket.Handler(HandleWS).ServeHTTP(w, r)
	}))
	mux.HandleFunc("/api/accounts", requireDashboardAuth(cfgMgr, handleAPIAccounts))
	mux.HandleFunc("/api/accounts/toggle", requireDashboardAuth(cfgMgr, handleAPIAccountToggle))
	mux.HandleFunc("/api/refresh-quotas", requireDashboardAuth(cfgMgr, handleAPIRefreshQuotas))
	mux.HandleFunc("/api/warmup", requireDashboardAuth(cfgMgr, handleAPIWarmup))
	mux.HandleFunc("/api/warmup-all", requireDashboardAuth(cfgMgr, handleAPIWarmupAll))
	mux.HandleFunc("/api/logs", requireDashboardAuth(cfgMgr, func(w http.ResponseWriter, r *http.Request) { handleAPILogs(cfgMgr, w, r) }))
	mux.HandleFunc("/api/sessions", requireDashboardAuth(cfgMgr, handleAPISessions))
	mux.HandleFunc("/api/session", requireDashboardAuth(cfgMgr, handleAPISessionDetail))
	mux.HandleFunc("/api/model-catalog", requireDashboardAuth(cfgMgr, handleAPIModelCatalog))
	mux.HandleFunc("/api/keys", requireDashboardAuth(cfgMgr, func(w http.ResponseWriter, r *http.Request) {
		handleAPIKeys(cfgMgr, w, r)
	}))
	mux.HandleFunc("/oauth/login", requireDashboardAuth(cfgMgr, handleOAuthLogin))
	mux.HandleFunc("/oauth/callback", requireDashboardAuth(cfgMgr, handleOAuthCallback))
}

// handleAPIModelCatalog exposes the real per-model limits (context window,
// max output tokens, thinking budget) fetched from Google's own CloudCode
// API — used by translator.go/context_window.go for clamping/truncation.
func handleAPIModelCatalog(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(globalModelCatalog.GetAll())
}

// accountsDashboardStaleAfter is how old an account's quota snapshot can get
// before a dashboard poll opportunistically refreshes it in the background.
// Deliberately NOT a standalone server-side ticker: nothing needs this data
// fresh except a human actually looking at the dashboard right now, and the
// dashboard already polls this endpoint every 15s on its own — piggybacking
// here means zero extra Google API calls while nobody has the page open,
// same spirit as applyRateLimitHeaders' 45s trigger for active traffic, just
// for accounts idle traffic never touches.
const accountsDashboardStaleAfter = 2 * time.Minute

func handleAPIAccounts(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodDelete {
		email := r.URL.Query().Get("email")
		if email == "" {
			http.Error(w, `{"error":"missing required query param: email"}`, http.StatusBadRequest)
			return
		}
		if err := globalAccountManager.DeleteAccount(email); err != nil {
			http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(fmt.Sprintf(`{"success":true,"email":%q}`, email)))
		return
	}

	accounts := globalAccountManager.GetAccounts()
	for _, acc := range accounts {
		globalAccountManager.TriggerAsyncQuotaRefresh(acc.Email, accountsDashboardStaleAfter)
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(accounts)
}

func handleAPIAccountToggle(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	email := r.URL.Query().Get("email")
	disabledStr := r.URL.Query().Get("disabled")
	if email == "" || disabledStr == "" {
		http.Error(w, `{"error":"missing email or disabled parameter"}`, http.StatusBadRequest)
		return
	}
	disabled := (disabledStr == "true" || disabledStr == "1")
	if err := globalAccountManager.ToggleAccountDisabled(email, disabled); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(fmt.Sprintf(`{"success":true,"email":%q,"disabled":%v}`, email, disabled)))
}

func handleAPILogs(cfgMgr *ConfigManager, w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodDelete {
		globalLogStore.Clear()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"cleared"}`))
		return
	}

	key := r.URL.Query().Get("key")
	account := r.URL.Query().Get("account")
	session := r.URL.Query().Get("session")
	agent := r.URL.Query().Get("agent")
	onlyErrors := r.URL.Query().Get("errors") == "true" || r.URL.Query().Get("errors") == "1"
	limitStr := r.URL.Query().Get("limit")

	limit := 100
	if limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 {
			limit = l
		}
	}

	logs, stats := globalLogStore.Query(key, account, session, agent, onlyErrors, limit)

	accounts := globalAccountManager.GetAccounts()
	tenants := cfgMgr.GetAllTenants()
	stats.TotalAccounts = len(accounts)
	stats.ActiveKeys = len(tenants)

	resp := map[string]interface{}{
		"logs":  logs,
		"stats": stats,
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func handleAPISessions(w http.ResponseWriter, r *http.Request) {
	limitStr := r.URL.Query().Get("limit")
	limit := 50
	if limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 {
			limit = l
		}
	}
	sessions := globalLogStore.QuerySessions(limit)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(sessions)
}

// handleAPISessionDetail returns the full timeline for exactly one session
// (GET /api/session?id=<session_id>): aggregate summary plus every request
// logged under it, oldest first. Built for MCP-side verification that a
// given session actually routed/behaved as expected — /api/logs's 500-row
// cap and newest-first order make it unsuitable for checking a specific
// session's complete, in-order behavior.
func handleAPISessionDetail(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	w.Header().Set("Content-Type", "application/json")
	if id == "" {
		http.Error(w, `{"error":"missing required query param: id"}`, http.StatusBadRequest)
		return
	}

	summary, requests := globalLogStore.GetSessionDetail(id)
	if summary == nil {
		w.WriteHeader(http.StatusNotFound)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"error":      "no logged requests for this session_id",
			"session_id": id,
		})
		return
	}

	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"summary":  summary,
		"requests": requests,
	})
}

func handleAPIRefreshQuotas(w http.ResponseWriter, r *http.Request) {
	go globalAccountManager.RefreshAllQuotas()
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(`{"status":"refresh_started"}`))
}

func handleAPIWarmup(w http.ResponseWriter, r *http.Request) {
	email := r.URL.Query().Get("email")
	if email == "" {
		http.Error(w, "missing email", http.StatusBadRequest)
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	msg, err := globalAccountManager.WarmupSingleAccount(ctx, email)
	w.Header().Set("Content-Type", "application/json")
	if err != nil {
		w.WriteHeader(http.StatusBadGateway)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"success": false, "error": err.Error()})
		return
	}
	_ = json.NewEncoder(w).Encode(map[string]interface{}{"success": true, "message": msg})
}

// handleAPIWarmupAll warms up every non-disabled account whose Gemini weekly
// window sits at 100% remaining (never activated) — the dashboard's single
// "warm up stale accounts" button, replacing the old per-row warmup button.
func handleAPIWarmupAll(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	warmed, errs := globalAccountManager.WarmupAllFreshAccounts(ctx)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"warmed":  warmed,
		"errors":  errs,
		"count":   len(warmed),
	})
}

func handleAPIKeys(cfgMgr *ConfigManager, w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodPost {
		var req struct {
			Name          string `json:"name"`
			StickyAccount string `json:"sticky_account"`
			Strategy      string `json:"strategy"`
			Model         string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		key, err := cfgMgr.CreateAPIKey(req.Name, req.StickyAccount, req.Strategy, req.Model)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"key": key, "name": req.Name})
		return
	}

	if r.Method == http.MethodDelete {
		key := r.URL.Query().Get("key")
		_ = cfgMgr.DeleteAPIKey(key)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"deleted"}`))
		return
	}

	tenants := cfgMgr.GetAllTenants()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(tenants)
}

func handleOAuthLogin(w http.ResponseWriter, r *http.Request) {
	state := newOAuthState()
	authURL := BuildGoogleAuthURL(state)
	http.Redirect(w, r, authURL, http.StatusTemporaryRedirect)
}

func handleOAuthCallback(w http.ResponseWriter, r *http.Request) {
	state := r.URL.Query().Get("state")
	if !consumeOAuthState(state) {
		// Classic OAuth CSRF: without this check, a forged callback URL
		// (any attacker-controlled "code") would be exchanged and bound
		// into this gateway's account pool as if the admin initiated it.
		http.Error(w, "Invalid or expired OAuth state", http.StatusBadRequest)
		return
	}

	code := r.URL.Query().Get("code")
	if code == "" {
		http.Error(w, "Missing authorization code", http.StatusBadRequest)
		return
	}

	acc, err := ExchangeAuthCode(code)
	if err != nil {
		http.Error(w, fmt.Sprintf("OAuth exchange error: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	html := fmt.Sprintf(`<!DOCTYPE html>
<html>
<head><title>Авторизация успешна</title><meta charset="utf-8"></head>
<body style="font-family:sans-serif;background:#090d16;color:#f8fafc;padding:40px;text-align:center;">
  <h2 style="color:#10b981;">✓ Аккаунт успешно добавлен!</h2>
  <p>Email: <b>%s</b></p>
  <p><a href="/" style="display:inline-block;padding:8px 16px;background:#3b82f6;color:white;text-decoration:none;border-radius:6px;font-size:13px;font-weight:600;">Вернуться в консоль</a></p>
</body>
</html>`, acc.Email)
	_, _ = w.Write([]byte(html))
}

func handleDashboard(cfgMgr *ConfigManager, w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write([]byte(dashboardHTML))
}

//go:embed dashboard.html
var dashboardHTML string
