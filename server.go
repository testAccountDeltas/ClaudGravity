package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

var (
	// Chrome-fingerprinted transport for every outbound call to Google
	// (inference, OAuth, quotas, model catalog, web-search grounding): Go's
	// own crypto/tls ClientHello doesn't match any real browser or the
	// genuine Antigravity client, which is a JA3/JA4-level signal Google
	// could filter on before the request content is even inspected.
	httpClient = &http.Client{
		Transport: newFingerprintedRoundTripper(),
	}
	reqCounter uint64
	reqMu      sync.Mutex
)

const (
	sessionBumpStaleAfter = 2 * time.Hour
	sessionBumpSweepEvery = 30 * time.Minute
)

// sessionBumpMap remembers, per session, the last session-id "bump" that
// got a long (>1M-token) conversation past the upstream session token-wall.
// Without this, bump always restarted at 0 on every new HTTP request, so
// every turn after the wall was first hit paid for a guaranteed-failing
// bump=0 attempt before retrying with the bump that actually works.
var (
	sessionBumpMu   sync.Mutex
	sessionBumpMap  = make(map[string]int)
	sessionBumpSeen = make(map[string]time.Time)
)

func init() {
	go func() {
		ticker := time.NewTicker(sessionBumpSweepEvery)
		defer ticker.Stop()
		for range ticker.C {
			sweepStaleSessionBumps()
		}
	}()
}

func sweepStaleSessionBumps() {
	sessionBumpMu.Lock()
	defer sessionBumpMu.Unlock()
	cutoff := time.Now().Add(-sessionBumpStaleAfter)
	for sessionID, t := range sessionBumpSeen {
		if t.Before(cutoff) {
			delete(sessionBumpSeen, sessionID)
			delete(sessionBumpMap, sessionID)
		}
	}
}

func getSessionBump(sessionID string) int {
	sessionBumpMu.Lock()
	defer sessionBumpMu.Unlock()
	return sessionBumpMap[sessionID]
}

func setSessionBump(sessionID string, bump int) {
	if sessionID == "" {
		return
	}
	sessionBumpMu.Lock()
	defer sessionBumpMu.Unlock()
	sessionBumpMap[sessionID] = bump
	sessionBumpSeen[sessionID] = time.Now()
}

// recoverMiddleware stops a panic in any single request from crashing the
// whole process — previously nothing in this codebase called recover(), so
// one bad request (malformed body, a nil-pointer edge case in translation,
// etc.) would take down every in-flight request and the whole gateway.
func recoverMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				LogGateway("[SuperRelay Gateway] PANIC recovered on %s %s: %v", r.Method, r.URL.Path, rec)
				writeAnthropicError(w, http.StatusInternalServerError, "internal_error", "internal gateway error")
			}
		}()
		next.ServeHTTP(w, r)
	})
}

var requestIDCounter uint64

// nextRequestID mints an id in the shape of a real Anthropic request-id
// (req_<hex>) for the request-id response header below — never parsed by
// clients, just expected to be present and look like one.
func nextRequestID() string {
	n := atomic.AddUint64(&requestIDCounter, 1)
	h := sha256.Sum256([]byte(fmt.Sprintf("%d-%d", time.Now().UnixNano(), n)))
	return "req_" + hex.EncodeToString(h[:12])
}

// commonAPIMiddleware answers browser/Electron CORS preflight (OPTIONS)
// requests and stamps every response with anthropic-version/request-id —
// both expected unconditionally by the official Anthropic SDKs and by any
// web/Electron-based client (Claude Web Artifacts, Cursor's webview,
// LibreChat, OpenWebUI, browser extensions), none of which this gateway
// previously sent at all, leaving browser clients blocked entirely by the
// browser's own CORS policy before the request even reached this code.
func commonAPIMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("Access-Control-Allow-Origin", "*")
		h.Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, DELETE, PUT")
		h.Set("Access-Control-Allow-Headers", "Content-Type, Authorization, x-api-key, anthropic-version, anthropic-beta, x-claude-code-request-class, x-claude-code-agent-type, x-claude-code-agent-id, x-claude-code-prompt-id, x-session-id, x-claude-code-session-id")
		h.Set("Access-Control-Expose-Headers", "anthropic-version, request-id, retry-after, anthropic-ratelimit-requests-limit, anthropic-ratelimit-requests-remaining, anthropic-ratelimit-requests-reset, anthropic-ratelimit-tokens-limit, anthropic-ratelimit-tokens-remaining, anthropic-ratelimit-tokens-reset, anthropic-ratelimit-unified-5h-status, anthropic-ratelimit-unified-5h-utilization, anthropic-ratelimit-unified-5h-reset, anthropic-ratelimit-unified-7d-status, anthropic-ratelimit-unified-7d-utilization, anthropic-ratelimit-unified-7d-reset, anthropic-ratelimit-unified-status, anthropic-ratelimit-unified-representative-claim, anthropic-ratelimit-unified-reset")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		h.Set("anthropic-version", "2023-06-01")
		h.Set("request-id", nextRequestID())
		next.ServeHTTP(w, r)
	})
}

// writeAnthropicError writes an error body matching Anthropic's actual
// wire format ({"type":"error","error":{"type":...,"message":...}}) with
// Content-Type: application/json. http.Error always forces
// "Content-Type: text/plain; charset=utf-8" regardless of what's in the
// body, which is what every one of this gateway's JSON-shaped error
// responses was using — a client SDK that checks Content-Type before
// attempting to parse a 4xx/5xx body (as Anthropic's own TypeScript SDK
// does) never even tries to read the structured error out of it.
// anthropicErrorTypeForStatus maps an HTTP status to the error type real
// Anthropic SDKs key their auto-retry logic on. The one call site that used
// to hardcode "api_error" regardless of status (the final give-up after the
// account-rotation loop exhausts retries) meant a 429 (quota) or 503/502/504
// (upstream overloaded) got surfaced as a type SDKs don't know to retry —
// they'd hard-fail instead of backing off and trying again.
func anthropicErrorTypeForStatus(status int) string {
	switch status {
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	case http.StatusBadGateway, http.StatusServiceUnavailable, http.StatusGatewayTimeout:
		return "overloaded_error"
	case http.StatusBadRequest:
		return "invalid_request_error"
	case http.StatusUnauthorized:
		return "authentication_error"
	case http.StatusForbidden:
		return "permission_error"
	case http.StatusNotFound:
		return "not_found_error"
	case http.StatusRequestEntityTooLarge:
		return "request_too_large"
	default:
		return "api_error"
	}
}

// stripThoughtFieldsForPublicAPI removes {"thought":true,...} parts'
// thought/thoughtSignature keys before a request goes to the public Gemini
// Developer API, which documents "thought" as output-only on input contents
// and 400s with "Field 'thought' is not supported for input contents" if a
// replayed conversation still carries it (translator.go always emits it
// regardless of which upstream the request will hit).
func stripThoughtFieldsForPublicAPI(geminiBody []byte) []byte {
	if !gjson.GetBytes(geminiBody, "contents").Exists() {
		return geminiBody
	}
	var root map[string]interface{}
	if err := json.Unmarshal(geminiBody, &root); err != nil {
		return geminiBody
	}
	contents, ok := root["contents"].([]interface{})
	if !ok {
		return geminiBody
	}
	changed := false
	for _, c := range contents {
		cm, ok := c.(map[string]interface{})
		if !ok {
			continue
		}
		parts, ok := cm["parts"].([]interface{})
		if !ok {
			continue
		}
		for _, p := range parts {
			pm, ok := p.(map[string]interface{})
			if !ok {
				continue
			}
			if _, hasThought := pm["thought"]; hasThought {
				delete(pm, "thought")
				delete(pm, "thoughtSignature")
				changed = true
			}
		}
	}
	if !changed {
		return geminiBody
	}
	out, err := json.Marshal(root)
	if err != nil {
		return geminiBody
	}
	return out
}

func writeAnthropicError(w http.ResponseWriter, status int, errType, message string) {
	w.Header().Set("Content-Type", "application/json")
	// The real Anthropic API answers an overloaded backend with HTTP 529,
	// not 502/503/504 — some SDK retry/backoff logic keys off the literal
	// wire status code, not just the JSON "type" field, so passing
	// Google's own upstream gateway status through verbatim here left that
	// 529-specific path unreachable. The body's "type" still says
	// "overloaded_error" either way; only the status line changes.
	wireStatus := status
	if errType == "overloaded_error" {
		wireStatus = 529
	}
	if status == http.StatusTooManyRequests {
		w.Header().Set("Retry-After", "5")
	} else if errType == "overloaded_error" {
		w.Header().Set("Retry-After", "3")
	}
	w.WriteHeader(wireStatus)
	body, _ := json.Marshal(map[string]interface{}{
		"type": "error",
		"error": map[string]string{
			"type":    errType,
			"message": message,
		},
	})
	_, _ = w.Write(body)
}

func nextReqID() uint64 {
	reqMu.Lock()
	defer reqMu.Unlock()
	reqCounter++
	return reqCounter
}

func main() {
	cfgPath := "config.json"
	if len(os.Args) > 1 {
		cfgPath = os.Args[1]
	}

	cfgMgr, err := LoadConfig(cfgPath)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	InitLogger("superrelay.log")

	// Same "defined but never called" pattern as the model catalog below:
	// InitLogStore was never invoked anywhere, so globalLogStore.db stayed
	// nil and every Add()/Query()/QuerySessions() call silently no-opped —
	// the whole /api/logs and /api/sessions dashboard subsystem was dead.
	if err := InitLogStore(""); err != nil {
		LogGateway("[SuperRelay Gateway] Warning: request log DB init failed: %v", err)
	}

	// Initialize account manager from auths directory
	if err := InitAccountManager("auths"); err != nil {
		LogGateway("[SuperRelay Gateway] Notice: auths dir init: %v", err)
	}

	// Populate the real per-model catalog (context window / max output /
	// thinking budget ceilings) from Google's own CloudCode API. Without
	// this, translator.go's max_tokens clamping and context_window.go's
	// truncation both silently no-op (globalModelCatalog stays empty) —
	// this was previously defined but never actually called anywhere.
	if _, err := globalAccountManager.FetchAvailableModels(false); err != nil {
		LogGateway("[SuperRelay Gateway] Warning: initial model catalog fetch failed: %v", err)
	} else {
		LogGateway("[SuperRelay Gateway] Model catalog loaded (context window / output ceilings)")
	}
	go func() {
		for {
			time.Sleep(10 * time.Minute)
			if _, err := globalAccountManager.FetchAvailableModels(true); err != nil {
				LogGateway("[SuperRelay Gateway] Warning: periodic model catalog refresh failed: %v", err)
			}
		}
	}()

	// Initialize native MITM Certificate Authority for Anthropic host interception
	mitmCA, err := InitMitmCA("certs")
	if err != nil {
		LogGateway("[SuperRelay Gateway] Warning: MitmCA init failed: %v", err)
	} else {
		LogGateway("[SuperRelay Gateway] Native MITM CA ready: %s", mitmCA.GetCAPath())
	}
	globalMitmCA = mitmCA

	serverCfg := cfgMgr.GetServerConfig()

	mux := http.NewServeMux()
	RegisterUIHandlers(mux, cfgMgr)

	apiMux := commonAPIMiddleware(mux)

	mux.HandleFunc("/v1/messages", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			handleMessages(cfgMgr, w, r)
		} else {
			http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		}
	})
	mux.HandleFunc("/v1/messages/count_tokens", func(w http.ResponseWriter, r *http.Request) {
		handleCountTokens(cfgMgr, w, r)
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		handleModels(cfgMgr, w, r)
	})
	mux.HandleFunc("/v1/models/{id}", func(w http.ResponseWriter, r *http.Request) {
		handleModelByID(cfgMgr, w, r)
	})
	mux.HandleFunc("/v1/usage", func(w http.ResponseWriter, r *http.Request) {
		handleUsage(cfgMgr, w, r)
	})
	mux.HandleFunc("/api/oauth/usage", func(w http.ResponseWriter, r *http.Request) {
		handleOAuthUsage(cfgMgr, w, r)
	})
	mux.HandleFunc("/api/claude_cli/bootstrap", func(w http.ResponseWriter, r *http.Request) {
		HandleBootstrapSpoof(w, r)
	})
	mux.HandleFunc("/v1/embeddings", func(w http.ResponseWriter, r *http.Request) {
		handleEmbeddings(cfgMgr, w, r)
	})

	// Separate mux for TLS-intercepted anthropic.com/claude.ai traffic (see
	// HandleConnectTunnel). This must NOT be the same mux as above: that one
	// has "/" bound to the web dashboard, which would otherwise swallow and
	// 404 every intercepted path we don't explicitly fake (marketplaces,
	// mcp-registry, telemetry, first-run connectivity checks, ...). Only the
	// two hardcoded-bypass endpoints are answered locally; everything else
	// falls through to a real passthrough to the actual host.
	interceptedMux := http.NewServeMux()
	// Every route this gateway answers itself here (as opposed to the
	// catch-all PassthroughToReal below, which forwards verbatim to the
	// REAL api.anthropic.com and must keep its own upstream headers intact)
	// gets commonAPIMiddleware too — previously only the plain-HTTP/TLS
	// listeners had CORS/anthropic-version/request-id, so a client set up
	// with only HTTPS_PROXY (routed through this MITM'd path instead) got
	// none of them on otherwise-identical responses.
	interceptedMux.Handle("/api/oauth/usage", commonAPIMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handleOAuthUsage(cfgMgr, w, r)
	})))
	interceptedMux.Handle("/api/claude_cli/bootstrap", commonAPIMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		HandleBootstrapSpoof(w, r)
	})))
	// A client set up with only HTTPS_PROXY (no ANTHROPIC_BASE_URL override)
	// sends its real traffic as a CONNECT tunnel to api.anthropic.com, MITM'd
	// by mitmCA above — previously only the two routes above were answered
	// here, and /v1/messages itself fell through to PassthroughToReal, which
	// forwards straight to the real Anthropic server where this gateway's
	// own key is meaningless and gets a 401. Answer the same routes here
	// that the plain-HTTP/TLS listeners already serve.
	interceptedMux.Handle("/v1/messages", commonAPIMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			handleMessages(cfgMgr, w, r)
		} else {
			http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		}
	})))
	interceptedMux.Handle("/v1/messages/count_tokens", commonAPIMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handleCountTokens(cfgMgr, w, r)
	})))
	interceptedMux.Handle("/v1/models", commonAPIMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handleModels(cfgMgr, w, r)
	})))
	interceptedMux.Handle("/v1/models/{id}", commonAPIMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handleModelByID(cfgMgr, w, r)
	})))
	interceptedMux.HandleFunc("/", PassthroughToReal)

	proxySecret := cfgMgr.ProxySecret()

	// Combined HTTP + HTTPS CONNECT Proxy root handler
	rootHandler := recoverMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodConnect && mitmCA != nil {
			HandleConnectTunnel(w, r, mitmCA, interceptedMux, proxySecret)
			return
		}
		apiMux.ServeHTTP(w, r)
	}))

	addr := fmt.Sprintf(":%d", serverCfg.Port)
	srv := &http.Server{
		Addr:    addr,
		Handler: rootHandler,
	}

	LogGateway("[SuperRelay Gateway] Listening on http://localhost:%d (CONNECT proxy + plain API)", serverCfg.Port)
	LogGateway("[SuperRelay Gateway] Panic Watchdog: %ds, TTFT Timeout: %ds", serverCfg.PanicTimeoutSeconds, serverCfg.TTFTTimeoutSeconds)
	LogGateway("[SuperRelay Gateway] Dashboard password: %s (Basic Auth, user: admin)", cfgMgr.DashboardPassword())
	LogGateway("[SuperRelay Gateway] Proxy-Authorization secret (set as 'Bearer <secret>' or via http://<secret>:x@host:port as HTTPS_PROXY): %s", proxySecret)

	// Second listener: plain HTTPS for ANTHROPIC_BASE_URL, using a
	// gateway-identity certificate signed by the same CA the client already
	// trusts for MITM interception. Kept on a separate port from the CONNECT
	// proxy above, since a CONNECT request line is sent as plain HTTP by
	// convention — wrapping that same port in TLS would break proxying.
	var tlsSrv *http.Server
	if mitmCA != nil {
		tlsPort := serverCfg.TLSPort
		gwCert, err := mitmCA.GetCertificate(serverCfg.PublicHost)
		if err != nil {
			LogGateway("[SuperRelay Gateway] Warning: could not mint gateway TLS cert: %v", err)
		} else {
			tlsSrv = &http.Server{
				Addr:      fmt.Sprintf(":%d", tlsPort),
				Handler:   recoverMiddleware(apiMux),
				TLSConfig: &tls.Config{Certificates: []tls.Certificate{gwCert}},
			}
			go func() {
				LogGateway("[SuperRelay Gateway] Listening on https://localhost:%d (TLS API, use as ANTHROPIC_BASE_URL)", tlsPort)
				if err := tlsSrv.ListenAndServeTLS("", ""); err != nil && err != http.ErrServerClosed {
					LogGateway("[SuperRelay Gateway] TLS server error: %v", err)
				}
			}()
		}
	}

	// Graceful shutdown: previously only the plain HTTP/CONNECT server (srv)
	// was shut down — the TLS API listener (tlsSrv, port 8900, what
	// ANTHROPIC_BASE_URL actually points at) and the SQLite log store kept
	// running until the process was killed outright, leaving in-flight TLS
	// requests cut off mid-response and the WAL file uncheckpointed.
	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
		<-sigChan
		LogGateway("[SuperRelay Gateway] Shutting down...")
		// 5s was too tight for this gateway's own normal traffic: a single
		// heavy ModeAuto generation routinely runs 30-170s+ (seen live in
		// superrelay.log all session), so a restart during one would force-
		// close the SSE stream mid-answer instead of letting it finish.
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_ = srv.Shutdown(ctx)
		if tlsSrv != nil {
			_ = tlsSrv.Shutdown(ctx)
		}
		globalLogStore.Close()
	}()

	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		LogGateway("Server error: %v", err)
	}
}

func handleMessages(cfgMgr *ConfigManager, w http.ResponseWriter, r *http.Request) {
	rid := nextReqID()
	t0 := time.Now()

	// 1. Resolve Tenant by API key. Claude Code's own OAuth/subscription
	// flow (what this gateway's launchers all use) sends Authorization:
	// Bearer — but every official Anthropic SDK (Python, TypeScript) and
	// any client configured with a plain ANTHROPIC_API_KEY instead sends
	// x-api-key by default, which this never checked at all.
	clientKey := r.Header.Get("x-api-key")
	if clientKey == "" {
		authHeader := r.Header.Get("Authorization")
		clientKey = strings.TrimPrefix(authHeader, "Bearer ")
	}
	tenant, ok := cfgMgr.GetTenant(clientKey)
	if !ok {
		LogGateway("[#%d] Unauthorized: invalid or missing Bearer token", rid)
		writeAnthropicError(w, http.StatusUnauthorized, "authentication_error", "Invalid or missing API key")
		globalLogStore.Add(RequestLog{
			Timestamp: t0,
			Key:       clientKey,
			KeyMasked: MaskKey(clientKey),
			Account:   "-",
			SessionID: "-",
			AgentKey:  "main",
			AgentType: "main",
			Model:     "-",
			Status:    http.StatusUnauthorized,
			Error:     "Missing API key",
		})
		return
	}

	agentKey, agentType, subagentName, instanceID := ParseAgentInfo(r)
	reqClass := r.Header.Get("x-claude-code-request-class")

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "Failed to read request body")
		return
	}

	// 2. Derive Conversation Anchor for Session Affinity (Prompt Caching)
	headerSessionID := r.Header.Get("x-claude-code-session-id")
	if headerSessionID == "" {
		headerSessionID = r.Header.Get("x-session-id")
	}
	convAnchor := DeriveConversationAnchor(bodyBytes, headerSessionID)

	// 2.1 Fast-Path Intercept: Claude Code Native WebSearch Sub-Request
	if searchQuery, isSearch := IsClaudeCodeDirectWebSearch(bodyBytes); isSearch {
		var activeAcc *AccountRecord
		var googleToken string
		isAntigravity := true

		if tenant.GoogleAuthType != "api_key" || tenant.GoogleAPIKey == "" || tenant.GoogleAPIKey == "YOUR_GEMINI_API_KEY_HERE" {
			acc, err := globalSelector.SelectAccount(clientKey, convAnchor, tenant)
			if err == nil && acc != nil {
				activeAcc = acc
				tok, _ := globalAccountManager.RefreshAccessToken(acc.Email)
				googleToken = tok
			}
		}
		accLabel := "DirectAPIKey"
		if activeAcc != nil {
			accLabel = activeAcc.Email
		}
		LogGateway("[#%d] >> INTERCEPT Claude Code Direct WebSearch: query=%q account=%s", rid, searchQuery, accLabel)
		handleClaudeCodeDirectWebSearch(r.Context(), rid, t0, clientKey, accLabel, tenant, googleToken, isAntigravity, activeAcc, bodyBytes, searchQuery, w)
		return
	}

	// Derived before translation (geminiBody doesn't exist yet) so the
	// LoopGuard key below can be scoped to this one conversation: its only
	// fallback that would otherwise need geminiBody is a last-resort content
	// hash, and bodyBytes is an equally stable stand-in for that.
	sessionID := DeriveSessionID(r, bodyBytes, bodyBytes)

	// Native 3P model names (claude-sonnet-4-6 etc.) pass through
	// tiering.go's ResolveTieringAndEffort unchanged — fine when the
	// account serving this request has 3P quota, but free-tier accounts
	// commonly sit at 0% there while Gemini itself is wide open. Account
	// selection for the real request happens further below (translation
	// has to run first — it decides the model that selection routes on),
	// so this peeks the conversation's already-sticky account, if any, to
	// catch the common case: an established multi-turn chat that's about
	// to land back on the same exhausted account it was already using.
	reqModel := gjson.GetBytes(bodyBytes, "model").String()
	if reqModel == "claude-sonnet-4-6" || reqModel == "claude-opus-4-6-thinking" || reqModel == "gpt-oss-120b-medium" {
		if peekEmail := globalSelector.GetAffinityAccount(convAnchor); peekEmail != "" {
			if peekAcc, ok := globalAccountManager.GetAccount(peekEmail); ok && isNative3PModelExhausted(peekAcc) {
				if patched, err := sjson.SetBytes(bodyBytes, "model", "gemini-3.8-flash-tiered"); err == nil {
					LogGateway("[#%d] !! Native 3P model %s has exhausted quota on sticky account %s, remapping to gemini-3.8-flash-tiered", rid, reqModel, peekEmail)
					bodyBytes = patched
				}
			}
		}
	}

	// 3. Full Orchestrator Translation: Tiering, LoopGuard, Middle-Truncation 64KB, Sanitizer
	geminiBody, modelName, budget, wasShrunk, syntheticReasonTools, err := ConvertClaudeRequestToGemini(bodyBytes, tenant, reqClass, agentKey, sessionID, instanceID)
	if err != nil {
		LogGateway("[#%d] Translation error: %v", rid, err)
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", fmt.Sprintf("Translation error: %v", err))
		return
	}

	isStream := gjson.GetBytes(bodyBytes, "stream").Bool()
	serverCfg := cfgMgr.GetServerConfig()
	// Auto-detect and record session working directory for Edit repair.
	// Was passing .Raw (undecoded source JSON bytes, escape sequences
	// un-decoded — a real newline is still the literal two characters
	// '\' 'n', not a newline byte) into a regex that stops at '\r'/'\n':
	// it never matched a real line break and ran on to capture garbage
	// from the rest of the JSON structure. ExtractSystemPromptText decodes
	// properly via gjson's .String() first.
	if cwd := ExtractCWDFromSystemPrompt(ExtractSystemPromptText(bodyBytes)); cwd != "" {
		RecordSessionCWD(sessionID, cwd)
	}
	RecordSessionFileSnapshots(sessionID, bodyBytes)

	hasSafeguards := gjson.GetBytes(bodyBytes, "safeguards").Exists()
	flusher, hasFlusher := w.(http.Flusher)
	isStreaming := isStream && hasFlusher

	// 4. Pre-stream Failover Loop (up to 3 attempts with account rotation)
	maxAttempts := 3
	excludedAccounts := make(map[string]bool)
	var lastErr error
	var lastStatus int = http.StatusBadGateway

	for attempt := 0; attempt < maxAttempts; attempt++ {
		var activeAcc *AccountRecord
		var googleToken string
		isAntigravity := true

		if tenant.GoogleAuthType == "api_key" && tenant.GoogleAPIKey != "" && tenant.GoogleAPIKey != "YOUR_GEMINI_API_KEY_HERE" {
			isAntigravity = false
		} else {
			acc, err := globalSelector.SelectAccountExcluding(clientKey, convAnchor, tenant, excludedAccounts)
			if err != nil {
				lastErr = err
				break
			}
			activeAcc = acc
			tok, err := globalAccountManager.RefreshAccessToken(acc.Email)
			if err != nil {
				LogGateway("[#%d] !! Token refresh failed for %s: %v (rotating account)", rid, acc.Email, err)
				excludedAccounts[acc.Email] = true
				lastErr = err
				continue
			}
			googleToken = tok

			// The sticky-affinity peek above the account-selection loop only
			// catches an established multi-turn conversation whose sticky
			// account happens to be known; a brand-new conversation's first
			// turn has no affinity entry yet, so it slipped through to
			// whichever account SelectAccountExcluding actually picked. If
			// THAT account's 3P quota is exhausted, sending the native
			// passthrough model anyway gets a 429 — and the 429 handler
			// below cools the whole account down for 15 minutes, locking it
			// out of its still-100%-free Gemini quota too. Catch it here,
			// against the account that's actually about to be used.
			if (modelName == "claude-sonnet-4-6" || modelName == "claude-opus-4-6-thinking" || modelName == "gpt-oss-120b-medium") && isNative3PModelExhausted(activeAcc) {
				LogGateway("[#%d] !! Native 3P model %s has exhausted quota on selected account %s, remapping to gemini-3.8-flash-tiered", rid, modelName, acc.Email)
				modelName = "gemini-3.8-flash-tiered"
			}
		}

		accLabel := "DirectAPIKey"
		projectID := "aicode-consumers"
		if activeAcc != nil {
			accLabel = activeAcc.Email
			if activeAcc.ProjectID != "" {
				projectID = activeAcc.ProjectID
			}
		}

		// Non-Antigravity upstream URL is fixed for the whole attempt; for
		// Antigravity it's rebuilt per host inside the retry loop below.
		var directURL string
		if !isAntigravity {
			// tiering.go's model names (gemini-3.8-flash-tiered/-low/-medium/
			// -high) are Antigravity/CloudCode-internal aliases — the public
			// Gemini Developer API this branch calls doesn't recognize them
			// and 404s. Only reachable once a tenant actually configures a
			// real google_api_key (both deployments currently still have
			// the config.example.json placeholder there), but worth fixing
			// correctly since this is a published, reusable gateway.
			if strings.HasPrefix(modelName, "gemini-3.8-flash") {
				modelName = "gemini-2.5-flash"
			}
			baseURL := tenant.GoogleBaseURL
			if baseURL == "" {
				baseURL = "https://generativelanguage.googleapis.com"
			}
			if isStreaming {
				directURL = fmt.Sprintf("%s/v1beta/models/%s:streamGenerateContent?alt=sse", baseURL, modelName)
			} else {
				directURL = fmt.Sprintf("%s/v1beta/models/%s:generateContent", baseURL, modelName)
			}
			if tenant.GoogleAPIKey != "" {
				if strings.Contains(directURL, "?") {
					directURL += "&key=" + tenant.GoogleAPIKey
				} else {
					directURL += "?key=" + tenant.GoogleAPIKey
				}
			}
		}

		// Combined host-fallback + session-bump retry loop, same account,
		// same conversation, two independent recovery strategies for two
		// distinct upstream signals (both confirmed real via measurement in
		// the clirelay project, which solves the identical problem for the
		// identical upstream):
		//
		//  - Location gate (400 "User location is not supported for the API
		//    use"): a verdict on which EXIT HOST answered, not on the
		//    request. Move to the next host in antigravityHosts and resend
		//    unchanged. Production is deliberately excluded from that list —
		//    it answers free-tier accounts with 429 RESOURCE_EXHAUSTED on
		//    every single request, and a 429 trips this gateway's own
		//    account cooldown, poisoning requests that follow.
		//  - Session token-wall (400 "...exceeds the maximum number of
		//    tokens...", ~1,048,576): a verdict on the upstream session
		//    cache bucket, not on the conversation. context_window.go's
		//    proactive truncation targets each model's real ceiling, but
		//    it's an estimate; if it still undershoots, bump the
		//    deterministic session id instead of losing history or burning
		//    an account-rotation slot on it.
		// Each bump just retries the same account with a fresh session
		// bucket -- it costs a lost prefix cache on that one retry, not an
		// account-rotation slot, so there is little downside to allowing
		// more of them. 2 ran out within 1-2 hours of a long, subagent-
		// heavy session hitting the wall repeatedly, forcing the request
		// to fall through to full account rotation instead.
		const maxSessionBumps = 8
		const maxLocationGatePasses = 2 // full passes over antigravityHosts
		hostIdx := 0
		locationGatePasses := 0
		// Start from this session's last-known-good bump instead of 0: once a
		// long (>1M-token) conversation has hit the session token-wall once,
		// every later turn would otherwise repeat the same failing bump=0
		// attempt first, burning a wasted round-trip on every single turn for
		// the rest of the conversation instead of just the first one.
		bump := getSessionBump(sessionID)
		var success bool
		for {
			upstreamURL := directURL
			if isAntigravity {
				host := antigravityHosts[hostIdx]
				if isStreaming {
					upstreamURL = host + "/v1internal:streamGenerateContent?alt=sse"
				} else {
					upstreamURL = host + "/v1internal:generateContent"
				}
			}

			var finalPayload []byte
			if isAntigravity {
				// Google's Antigravity session cache is shared and capped
				// (~1,048,576 tokens) per sessionId. sessionID alone is the
				// terminal-wide x-claude-code-session-id, identical across
				// every parallel subagent spawned from the same orchestrator
				// turn — without qualifying by instanceID, N concurrent
				// subagents' contexts all land in the same server-side
				// session bucket and collectively blow the wall even though
				// each one individually fits comfortably.
				antigravitySessionKey := sessionID
				if instanceID != "" {
					antigravitySessionKey = sessionID + ":" + instanceID
				}
				finalPayload = WrapAntigravityPayload(modelName, geminiBody, projectID, bump, antigravitySessionKey)
			} else {
				// The public Gemini Developer API documents "thought" as
				// output-only; translator.go always emits replayed
				// extended-thinking history as {"thought":true,...} parts
				// regardless of which upstream will see them, so a history
				// containing any thinking block 400s here with "Field
				// 'thought' is not supported for input contents" unless
				// stripped first.
				finalPayload = stripThoughtFieldsForPublicAPI(geminiBody)
			}

			LogGateway("[#%d] >> POST /v1/messages [attempt %d/%d, session-gen %d] key=%s account=%s model=%s thinkBudget=%d stream=%v bytes=%d [%s]",
				rid, attempt+1, maxAttempts, bump, tenant.Name, accLabel, modelName, budget, isStreaming, len(geminiBody), agentKey)

			if isStreaming {
				success, lastStatus, lastErr = executeStreamingWithWatchdog(r.Context(), rid, t0, activeAcc, clientKey, accLabel, sessionID, tenant, googleToken, isAntigravity, serverCfg, upstreamURL, finalPayload, modelName, agentKey, agentType, subagentName, instanceID, wasShrunk, hasSafeguards, w, flusher, bodyBytes, syntheticReasonTools)
			} else {
				success, lastStatus, lastErr = executeNonStreaming(r.Context(), rid, t0, activeAcc, clientKey, accLabel, sessionID, tenant, googleToken, isAntigravity, serverCfg, upstreamURL, finalPayload, modelName, agentKey, agentType, subagentName, instanceID, wasShrunk, hasSafeguards, w, bodyBytes, syntheticReasonTools)
			}

			if success || !isAntigravity {
				break
			}
			errText := fmt.Sprint(lastErr)

			if isAntigravityLocationGateError(lastStatus, errText) {
				hostIdx++
				if hostIdx >= len(antigravityHosts) {
					hostIdx = 0
					locationGatePasses++
				}
				if locationGatePasses >= maxLocationGatePasses {
					LogGateway("[#%d] !! Location gate exhausted all hosts %d time(s) on %s, giving up on this account", rid, locationGatePasses, accLabel)
					break
				}
				LogGateway("[#%d] !! Location gate on %s, retrying on next host (%s)", rid, accLabel, antigravityHosts[hostIdx])
				continue
			}

			if isSessionTokenWallError(lastStatus, errText) {
				if bump >= maxSessionBumps {
					break
				}
				bump++
				setSessionBump(sessionID, bump)
				LogGateway("[#%d] !! Session token-wall (~1,048,576) hit on %s, bumping to a fresh session bucket and retrying same account", rid, accLabel)
				continue
			}

			break
		}

		if success {
			if activeAcc != nil && convAnchor != "" {
				globalSelector.RememberAffinity(convAnchor, activeAcc.Email)
			}
			return
		}

		// If failure occurred before headers were sent, we can safely retry on another account
		if activeAcc != nil {
			excludedAccounts[activeAcc.Email] = true
			if lastStatus == 429 || strings.Contains(fmt.Sprint(lastErr), "RESOURCE_EXHAUSTED") {
				globalAccountManager.MarkAccountCooldown(activeAcc.Email, 15*time.Minute)
			} else if lastStatus == http.StatusForbidden {
				// 403 means Google itself has rejected this account (banned,
				// suspended, region-gated) — not a transient rate limit.
				// excludedAccounts above only keeps it out of THIS request's
				// retry loop; without also unsticking it, every future
				// request for the same conversation keeps getting routed
				// right back to it by Session Affinity until a human
				// notices and disables it by hand.
				globalAccountManager.MarkAccountCooldown(activeAcc.Email, 60*time.Minute)
				globalSelector.UnstickAccount(activeAcc.Email)
			}
		}

		if !isAntigravity {
			break // No multiple accounts to rotate for direct API key
		}

		LogGateway("[#%d] !! Pre-stream failover: attempt %d failed (status=%d, err=%v), retrying on next account...", rid, attempt+1, lastStatus, lastErr)
		time.Sleep(100 * time.Millisecond) // brief jitter before retry
	}

	// If all attempts failed and nothing was sent yet
	if lastErr != nil {
		LogGateway("[#%d] !! All pre-stream failover attempts failed: %v", rid, lastErr)
		writeAnthropicError(w, lastStatus, anthropicErrorTypeForStatus(lastStatus), fmt.Sprintf("Upstream failed after retries: %v", lastErr))
	}
}

func executeStreamingWithWatchdog(
	baseCtx context.Context,
	rid uint64,
	t0 time.Time,
	activeAcc *AccountRecord,
	clientKey string,
	accLabel string,
	sessionID string,
	tenant TenantConfig,
	bearerToken string,
	isAntigravity bool,
	serverCfg ServerConfig,
	upstreamURL string,
	geminiBody []byte,
	modelName string,
	agentKey string,
	agentType string,
	subagentName string,
	instanceID string,
	wasShrunk bool,
	hasSafeguards bool,
	w http.ResponseWriter,
	flusher http.Flusher,
	claudeBodyBytes []byte,
	syntheticReasonTools map[string]bool,
) (bool, int, error) {
	panicTimeout := time.Duration(serverCfg.PanicTimeoutSeconds) * time.Second
	ttftTimeout := time.Duration(serverCfg.TTFTTimeoutSeconds) * time.Second

	// Idle-activity timeout instead of a flat total-duration ceiling: a huge
	// Write call can legitimately take minutes once Gemini actually starts
	// answering (confirmed live: 20K+ output tokens buffered as one lump,
	// ~120s), and a fixed "whole request must finish in N minutes" deadline
	// can't tell that apart from a genuinely hung connection — it either
	// kills real work or has to be set so generous it stops protecting
	// against hangs at all. This cancels only when upstream goes silent for
	// longer than idleTimeout, with maxStreamDuration as an absolute safety
	// net against a connection that trickles just enough to never go idle.
	idleTimeout := ttftTimeout
	maxStreamDuration := time.Duration(serverCfg.MaxStreamDurationSeconds) * time.Second
	if maxStreamDuration <= 0 {
		maxStreamDuration = 30 * time.Minute
	}

	// Deriving from baseCtx (the original http.Request's context) instead of
	// context.Background() means a client that disconnects (Ctrl+C, closed
	// terminal, dropped connection) now actually cancels the in-flight
	// upstream Google request immediately — net/http cancels a request's
	// Context on client disconnect — instead of letting the generation run
	// to completion anyway and silently burning the account's quota for a
	// response nobody will ever read.
	ctx, cancel := context.WithCancel(baseCtx)

	var lastActivity atomic.Int64
	lastActivity.Store(time.Now().UnixNano())
	touch := func() { lastActivity.Store(time.Now().UnixNano()) }

	idleWatchdogDone := make(chan struct{})
	go func() {
		defer close(idleWatchdogDone)
		ticker := time.NewTicker(2 * time.Second)
		defer ticker.Stop()
		start := time.Now()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				silentFor := time.Since(time.Unix(0, lastActivity.Load()))
				if silentFor > idleTimeout {
					LogGateway("[#%d] !! Idle watchdog: no bytes from upstream for %.0fs (limit %.0fs) -> cancelling", rid, silentFor.Seconds(), idleTimeout.Seconds())
					cancel()
					return
				}
				if time.Since(start) > maxStreamDuration {
					LogGateway("[#%d] !! Absolute stream-duration safety net hit (%.0fs) -> cancelling", rid, maxStreamDuration.Seconds())
					cancel()
					return
				}
			}
		}
	}()
	// MUST cancel() before waiting on idleWatchdogDone: the watchdog
	// goroutine only exits via ctx.Done() (or its own timeout firing), so a
	// plain `defer cancel()` registered before this would run AFTER this
	// defer (defers are LIFO) — deadlocking this function for up to
	// idleTimeout/maxStreamDuration on every single successful request. A
	// real deadlock of exactly this kind shipped: the response had already
	// been fully written and displayed client-side, but this handler never
	// returned, so Go's http.Server never sent the final chunk/closed the
	// connection — the client saw an unterminated stream and surfaced its
	// own "Waiting for API response... will retry" after already showing
	// the complete, correct reply.
	defer func() {
		cancel()
		<-idleWatchdogDone
	}()

	upReq, err := http.NewRequestWithContext(ctx, "POST", upstreamURL, bytes.NewReader(geminiBody))
	if err != nil {
		return false, http.StatusInternalServerError, fmt.Errorf("failed to create upstream request: %w", err)
	}
	upReq.Header.Set("Content-Type", "application/json")
	if bearerToken != "" {
		upReq.Header.Set("Authorization", "Bearer "+bearerToken)
	}
	if isAntigravity {
		upReq.Header.Set("User-Agent", AntigravityUserAgent)
		upReq.Header.Set("X-Goog-Api-Client", "google-cloud-sdk vscode_cloudshelleditor/0.1")
		upReq.Header.Set("Client-Metadata", `{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}`)
	}

	type upResult struct {
		resp *http.Response
		err  error
	}

	resChan := make(chan upResult, 1)
	go func() {
		resp, err := httpClient.Do(upReq)
		resChan <- upResult{resp: resp, err: err}
	}()

	var earlyHeadersSent bool
	var wMu sync.Mutex

	panicTicker := time.NewTicker(panicTimeout)
	defer panicTicker.Stop()

	var upResp *http.Response

	select {
	case res := <-resChan:
		touch()
		if res.err != nil {
			LogGateway("[#%d] !! Upstream connection failed: %v", rid, res.err)
			return false, http.StatusBadGateway, res.err
		}
		upResp = res.resp

	case <-panicTicker.C:
		LogGateway("[#%d] .. upstream TTFT >= %ds -> panicking: early SSE headers + keepalive ping", rid, serverCfg.PanicTimeoutSeconds)
		wMu.Lock()
		applyRateLimitHeaders(w.Header(), activeAcc, modelName)
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "close")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("event: ping\ndata: {\"type\":\"ping\"}\n\n"))
		flusher.Flush()
		earlyHeadersSent = true
		wMu.Unlock()

		for {
			select {
			case res := <-resChan:
				touch()
				if res.err != nil {
					LogGateway("[#%d] !! Upstream failed after panic: %v", rid, res.err)
					wMu.Lock()
					errPayload := fmt.Sprintf(`{"type":"error","error":{"type":"api_error","message":"Upstream failed after panic: %v"}}`, res.err)
					_, _ = w.Write([]byte(fmt.Sprintf("event: error\ndata: %s\n\n", errPayload)))
					flusher.Flush()
					wMu.Unlock()
					globalLogStore.Add(RequestLog{
						Timestamp:    t0,
						Key:          clientKey,
						KeyMasked:    MaskKey(clientKey),
						Account:      accLabel,
						SessionID:    sessionID,
						AgentKey:     agentKey,
						AgentType:    agentType,
						SubagentName: subagentName,
						InstanceID:   instanceID,
						Model:        modelName,
						TTFT:         time.Since(t0).Milliseconds(),
						TotalDur:     time.Since(t0).Milliseconds(),
						WasPanicked:  true,
						Shrunk:       wasShrunk,
						Error:        res.err.Error(),
						Status:       http.StatusBadGateway,
					})
					return true, http.StatusBadGateway, res.err
				}
				upResp = res.resp
				goto UpstreamReady

			case <-panicTicker.C:
				wMu.Lock()
				_, _ = w.Write([]byte("event: ping\ndata: {\"type\":\"ping\"}\n\n"))
				flusher.Flush()
				wMu.Unlock()

			case <-ctx.Done():
				LogGateway("[#%d] !! Upstream went idle/cancelled after panic (no response at all)", rid)
				wMu.Lock()
				errPayload := `{"type":"error","error":{"type":"api_error","message":"Upstream timed out: no response received"}}`
				_, _ = w.Write([]byte(fmt.Sprintf("event: error\ndata: %s\n\n", errPayload)))
				flusher.Flush()
				wMu.Unlock()
				globalLogStore.Add(RequestLog{
					Timestamp:    t0,
					Key:          clientKey,
					KeyMasked:    MaskKey(clientKey),
					Account:      accLabel,
					SessionID:    sessionID,
					AgentKey:     agentKey,
					AgentType:    agentType,
					SubagentName: subagentName,
					InstanceID:   instanceID,
					Model:        modelName,
					TTFT:         time.Since(t0).Milliseconds(),
					TotalDur:     time.Since(t0).Milliseconds(),
					WasPanicked:  true,
					Shrunk:       wasShrunk,
					Error:        "Upstream context timeout after panic",
					Status:       http.StatusGatewayTimeout,
				})
				return true, http.StatusGatewayTimeout, ctx.Err()
			}
		}
	}

UpstreamReady:
	defer upResp.Body.Close()

	if upResp.StatusCode != http.StatusOK {
		respBytes, _ := io.ReadAll(upResp.Body)
		LogGateway("[#%d] !! Upstream returned HTTP %d: %s", rid, upResp.StatusCode, string(respBytes))
		if upResp.StatusCode == 429 || strings.Contains(string(respBytes), "RESOURCE_EXHAUSTED") {
			globalAccountManager.MarkAccountCooldown(accLabel, 15*time.Minute)
		}
		if !earlyHeadersSent {
			// Early headers NOT sent yet: we can failover to another account!
			return false, upResp.StatusCode, fmt.Errorf("HTTP %d: %s", upResp.StatusCode, string(respBytes))
		}

		// Headers already sent: report SSE error block
		wMu.Lock()
		errPayload := fmt.Sprintf(`{"type":"error","error":{"type":%q,"message":"Upstream HTTP %d: %s"}}`, anthropicErrorTypeForStatus(upResp.StatusCode), upResp.StatusCode, string(respBytes))
		_, _ = w.Write([]byte(fmt.Sprintf("event: error\ndata: %s\n\n", errPayload)))
		flusher.Flush()
		wMu.Unlock()

		globalLogStore.Add(RequestLog{
			Timestamp:    t0,
			Key:          clientKey,
			KeyMasked:    MaskKey(clientKey),
			Account:      accLabel,
			SessionID:    sessionID,
			AgentKey:     agentKey,
			AgentType:    agentType,
			SubagentName: subagentName,
			InstanceID:   instanceID,
			Model:        modelName,
			TTFT:         time.Since(t0).Milliseconds(),
			TotalDur:     time.Since(t0).Milliseconds(),
			WasPanicked:  earlyHeadersSent,
			Shrunk:       wasShrunk,
			Error:        fmt.Sprintf("HTTP %d: %s", upResp.StatusCode, string(respBytes)),
			Status:       upResp.StatusCode,
		})
		return true, upResp.StatusCode, nil
	}

	wMu.Lock()
	if !earlyHeadersSent {
		applyRateLimitHeaders(w.Header(), activeAcc, modelName)
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "close")
		w.WriteHeader(http.StatusOK)
		flusher.Flush()
	}
	wMu.Unlock()

	ttft := time.Since(t0)
	// Gemini only reports usageMetadata at the very end of the stream, but
	// Anthropic's SDK expects message_start to carry a real (or at least
	// approximate) input_tokens count — its own message_delta handler only
	// ever updates output_tokens, so a client faithfully following that
	// contract never sees input_tokens corrected later and would otherwise
	// be stuck believing it was 0 for the whole conversation.
	estInputTokens := estimateApproxClaudeTokens(claudeBodyBytes)
	preferRussian := detectRussianConversation(claudeBodyBytes)
	streamer := NewSSEStreamer(w, flusher, modelName, sessionID, hasSafeguards, estInputTokens, preferRussian, syntheticReasonTools)

	err = StreamGeminiToClaude(streamer, upResp.Body, touch)
	totalDur := time.Since(t0)

	readCount := 0
	madeProgress := false
	for _, t := range streamer.toolNames {
		if t == "Read" {
			readCount++
		} else if t == "Write" || t == "Edit" || t == "NotebookEdit" {
			madeProgress = true
		}
	}
	// A turn with ZERO tool calls at all (readCount==0, no Write/Edit
	// either) that's still a substantial text answer IS real progress —
	// the agent explained something or finished. But textChars>200 used to
	// count as progress even when the SAME turn also called Read N times:
	// a model that narrates its own reading at length ("Let me carefully
	// check lines 100-200 to be sure...") wiped the loop-detection window
	// on every single read-loop turn purely by being verbose, which is
	// exactly the chatty-but-stuck pattern this guard exists to catch.
	if readCount == 0 && !madeProgress {
		madeProgress = streamer.textChars > 200
	}
	loopKey := agentKey + "|" + sessionID + "|" + instanceID
	readWindowSize, _ := readLoopLimits(agentKey)
	globalLoopGuard.RecordReadTurn(loopKey, readCount, madeProgress, readWindowSize)

	hasIgnoreTag := strings.Contains(streamer.lastText, OscIgnoreTag)
	globalLoopGuard.RecordToolCalls(loopKey, streamer.toolCallSignatures, hasIgnoreTag)

	var hitRate float64
	if streamer.inputTokens > 0 {
		hitRate = (float64(streamer.cacheTokens) / float64(streamer.inputTokens)) * 100.0
	}

	errStr := ""
	if err != nil {
		errStr = err.Error()
	}

	globalLogStore.Add(RequestLog{
		Timestamp:       t0,
		Key:             clientKey,
		KeyMasked:       MaskKey(clientKey),
		Account:         accLabel,
		SessionID:       sessionID,
		AgentKey:        agentKey,
		AgentType:       agentType,
		SubagentName:    subagentName,
		InstanceID:      instanceID,
		Model:           modelName,
		TTFT:            ttft.Milliseconds(),
		TotalDur:        totalDur.Milliseconds(),
		InTokens:        streamer.inputTokens,
		OutTokens:       streamer.outputTokens,
		CacheReadTokens: streamer.cacheTokens,
		CacheHitRate:    hitRate,
		Tools:           streamer.toolNames,
		WasPanicked:     earlyHeadersSent,
		Shrunk:          wasShrunk,
		Error:           errStr,
		Status:          http.StatusOK,
	})

	LogGateway("[#%d] << SSE done (%.1fs [ttft=%.1fs], in_tok=%d, out_tok=%d, cache_read=%d, tools=%v, err=%v)",
		rid, totalDur.Seconds(), ttft.Seconds(), streamer.inputTokens, streamer.outputTokens, streamer.cacheTokens, streamer.toolNames, err)

	return true, http.StatusOK, nil
}

func executeNonStreaming(
	baseCtx context.Context,
	rid uint64,
	t0 time.Time,
	activeAcc *AccountRecord,
	clientKey string,
	accLabel string,
	sessionID string,
	tenant TenantConfig,
	bearerToken string,
	isAntigravity bool,
	serverCfg ServerConfig,
	upstreamURL string,
	geminiBody []byte,
	modelName string,
	agentKey string,
	agentType string,
	subagentName string,
	instanceID string,
	wasShrunk bool,
	hasSafeguards bool,
	w http.ResponseWriter,
	claudeBodyBytes []byte,
	syntheticReasonTools map[string]bool,
) (bool, int, error) {
	// Unlike the streaming path (which has the panic/TTFT watchdog), this had
	// no deadline at all — a hung upstream could block the request forever.
	timeout := time.Duration(serverCfg.TTFTTimeoutSeconds+serverCfg.ReadTimeoutSeconds) * time.Second
	ctx, cancel := context.WithTimeout(baseCtx, timeout)
	defer cancel()

	upReq, err := http.NewRequestWithContext(ctx, "POST", upstreamURL, bytes.NewReader(geminiBody))
	if err != nil {
		return false, http.StatusInternalServerError, fmt.Errorf("failed to create upstream request: %w", err)
	}
	upReq.Header.Set("Content-Type", "application/json")
	if bearerToken != "" {
		upReq.Header.Set("Authorization", "Bearer "+bearerToken)
	}
	if isAntigravity {
		upReq.Header.Set("User-Agent", AntigravityUserAgent)
		upReq.Header.Set("X-Goog-Api-Client", "google-cloud-sdk vscode_cloudshelleditor/0.1")
		upReq.Header.Set("Client-Metadata", `{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}`)
	}

	resp, err := httpClient.Do(upReq)
	if err != nil {
		LogGateway("[#%d] !! Upstream non-streaming failed: %v", rid, err)
		return false, http.StatusBadGateway, err
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return false, http.StatusInternalServerError, fmt.Errorf("failed to read upstream body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		LogGateway("[#%d] !! Upstream non-streaming HTTP %d: %s", rid, resp.StatusCode, string(respBytes))
		return false, resp.StatusCode, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(respBytes))
	}

	geminiJSON := gjson.ParseBytes(respBytes)
	if respNode := geminiJSON.Get("response"); respNode.Exists() {
		geminiJSON = respNode
	}
	inTok := geminiJSON.Get("usageMetadata.promptTokenCount").Int()
	outTok := geminiJSON.Get("usageMetadata.candidatesTokenCount").Int()
	cacheTok := geminiJSON.Get("usageMetadata.cachedContentTokenCount").Int()

	// Previously only computed for the dashboard's own RequestLog entry
	// (CacheReadTokens below) and never surfaced to the client — Claude
	// Code's own /usage cache-hit-rate math silently saw 0% cache reads on
	// every non-streaming response.
	//
	// Gemini's promptTokenCount is the FULL input (cached + uncached), but
	// Anthropic's input_tokens means only the uncached delta — see the
	// matching fix in sse.go's FinishStream for the full explanation of why
	// reporting the full count here double-counts cache_read_input_tokens
	// on the client side.
	uncachedInTok := inTok - cacheTok
	if uncachedInTok < 0 {
		uncachedInTok = 0
	}
	// Same cache_creation_input_tokens approximation as sse.go's
	// FinishStream — see there for why Gemini gives us no real signal for
	// this and a cache-miss-but-large-prompt heuristic is the best stand-in.
	var cacheCreationTok int64
	if cacheTok == 0 && uncachedInTok > 1024 {
		cacheCreationTok = uncachedInTok
		uncachedInTok = 0
	}
	usageMap := map[string]interface{}{
		"input_tokens":  uncachedInTok,
		"output_tokens": outTok,
	}
	if cacheCreationTok > 0 {
		usageMap["cache_creation_input_tokens"] = cacheCreationTok
	}
	if cacheTok > 0 {
		usageMap["cache_read_input_tokens"] = cacheTok
	}

	// Previously always "end_turn" — a response that hit the output-token
	// ceiling looked identical to one that finished normally, so the client
	// had no signal to know its answer was truncated.
	stopReason := "end_turn"
	if geminiJSON.Get("candidates.0.finishReason").String() == "MAX_TOKENS" {
		stopReason = "max_tokens"
	}

	claudeResp := map[string]interface{}{
		"id":          fmt.Sprintf("msg_%d", time.Now().UnixNano()),
		"type":        "message",
		"role":        "assistant",
		"model":       modelName,
		"content":       []map[string]interface{}{},
		"usage":         usageMap,
		"stop_reason":   stopReason,
		"stop_sequence": nil,
	}

	var toolsCalled []string
	var toolCallIDs []string
	cand := geminiJSON.Get("candidates.0")
	if cand.Exists() {
		parts := cand.Get("content.parts")
		if parts.IsArray() {
			for _, p := range parts.Array() {
				if p.Get("thought").Bool() {
					if t := p.Get("text").String(); t != "" {
						block := map[string]interface{}{
							"type":     "thinking",
							"thinking": t,
						}
						if sig := p.Get("thoughtSignature").String(); sig != "" {
							block["signature"] = sig
						}
						claudeResp["content"] = append(claudeResp["content"].([]map[string]interface{}), block)
					}
				} else if t := p.Get("text").String(); t != "" {
					claudeResp["content"] = append(claudeResp["content"].([]map[string]interface{}), map[string]interface{}{
						"type": "text",
						"text": t,
					})
				} else if fc := p.Get("functionCall"); fc.Exists() {
					fname := fc.Get("name").String()
					argsRaw := fc.Get("args").Raw
					if argsRaw == "" {
						argsRaw = "{}"
					}
					argsRaw = coerceNumericEnumArgs(fname, argsRaw)
					if repairedArgs, ok := RepairEditToolCall(fname, argsRaw, sessionID); ok {
						argsRaw = repairedArgs
					}
					switch strings.ToLower(fname) {
					case "grep":
						if repaired, ok := RepairGrepToolCall(argsRaw); ok {
							argsRaw = repaired
						}
					case "glob":
						if repaired, ok := RepairGlobToolCall(argsRaw); ok {
							argsRaw = repaired
						}
					}
					if syntheticReasonTools[fname] {
						// Same synthetic-placeholder cleanup as sse.go's
						// HandleFunctionCall — see there for the full
						// explanation.
						if cleaned, err := sjson.Delete(argsRaw, "reason"); err == nil {
							argsRaw = cleaned
						}
					}
					var argsObj interface{}
					_ = json.Unmarshal([]byte(argsRaw), &argsObj)
					if argsObj == nil {
						// Malformed JSON (e.g. truncated mid-argument by a
						// MAX_TOKENS cutoff) leaves argsObj nil, which
						// serializes as "input": null — Anthropic's
						// Messages API requires tool_use.input to be an
						// object.
						argsObj = map[string]interface{}{}
					}
					toolsCalled = append(toolsCalled, fname)
					callID := nextToolCallID(fname)
					toolCallIDs = append(toolCallIDs, callID)
					setToolSignature(callID, p.Get("thoughtSignature").String())
					claudeResp["content"] = append(claudeResp["content"].([]map[string]interface{}), map[string]interface{}{
						"type":  "tool_use",
						"id":    callID,
						"name":  fname,
						"input": argsObj,
					})
					// Same MAX_TOKENS-wins-over-tool_use priority as sse.go's
					// FinishStream: a functionCall part being present
					// doesn't mean it arrived complete.
					if claudeResp["stop_reason"] != "max_tokens" {
						claudeResp["stop_reason"] = "tool_use"
					}
				}
			}
		}

		// Append Google Search Grounding sources if present
		if gm := cand.Get("groundingMetadata"); gm.Exists() {
			if chunks := gm.Get("groundingChunks"); chunks.IsArray() && len(chunks.Array()) > 0 {
				var sb strings.Builder
				if detectRussianConversation(claudeBodyBytes) {
					sb.WriteString("\n\n---\n**Источники Google Search:**\n")
				} else {
					sb.WriteString("\n\n---\n**Google Search Sources:**\n")
				}
				seen := make(map[string]bool)
				for _, c := range chunks.Array() {
					uri := c.Get("web.uri").String()
					title := c.Get("web.title").String()
					if uri != "" && !seen[uri] {
						seen[uri] = true
						if title == "" {
							title = uri
						}
						sb.WriteString(fmt.Sprintf("- [%s](%s)\n", title, uri))
					}
				}
				toolsCalled = append(toolsCalled, "googleSearch")
				claudeResp["content"] = append(claudeResp["content"].([]map[string]interface{}), map[string]interface{}{
					"type": "text",
					"text": sb.String(),
				})
			}
		}
	}

	if hasSafeguards {
		toolUsesMap := make(map[string]interface{})
		for _, tid := range toolCallIDs {
			toolUsesMap[tid] = map[string]interface{}{
				"type":    "evaluated",
				"outcome": "not_flagged",
			}
		}
		claudeResp["safeguard_results"] = []interface{}{
			map[string]interface{}{
				"type": "dangerous_tool_use",
				"status": map[string]interface{}{
					"type":      "available",
					"tool_uses": toolUsesMap,
				},
			},
		}
	}

	outJSON, _ := json.Marshal(claudeResp)
	applyRateLimitHeaders(w.Header(), activeAcc, modelName)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(outJSON)

	durMs := time.Since(t0).Milliseconds()
	var hitRate float64
	if inTok > 0 {
		hitRate = (float64(cacheTok) / float64(inTok)) * 100.0
	}

	globalLogStore.Add(RequestLog{
		Timestamp:       t0,
		Key:             clientKey,
		KeyMasked:       MaskKey(clientKey),
		Account:         accLabel,
		SessionID:       sessionID,
		AgentKey:        agentKey,
		AgentType:       agentType,
		SubagentName:    subagentName,
		InstanceID:      instanceID,
		Model:           modelName,
		TTFT:            durMs,
		TotalDur:        durMs,
		InTokens:        inTok,
		OutTokens:       outTok,
		CacheReadTokens: cacheTok,
		CacheHitRate:    hitRate,
		Tools:           toolsCalled,
		WasPanicked:     false,
		Shrunk:          wasShrunk,
		Status:          http.StatusOK,
	})

	LogGateway("[#%d] << Non-streaming done (%.1fs)", rid, time.Since(t0).Seconds())
	return true, http.StatusOK, nil
}

func handleCountTokens(cfgMgr *ConfigManager, w http.ResponseWriter, r *http.Request) {
	clientKey := r.Header.Get("x-api-key")
	if clientKey == "" {
		clientKey = strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	}
	if _, ok := cfgMgr.GetTenant(clientKey); !ok {
		writeAnthropicError(w, http.StatusUnauthorized, "authentication_error", "Invalid or missing API key")
		return
	}
	bodyBytes, _ := io.ReadAll(r.Body)
	estTokens := estimateApproxClaudeTokens(bodyBytes)
	resp := map[string]interface{}{
		"input_tokens": estTokens,
	}
	b, _ := json.Marshal(resp)
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(b)
}

// knownModels backs both /v1/models (list) and /v1/models/{id} (lookup) —
// previously only the list existed, and any client calling the per-model
// lookup (the official Anthropic SDKs do, to validate a model id) got a
// plain 404 instead of that model's metadata.
// "type": "model" (not "object": "model") and "created_at" match the real
// Anthropic Models API schema — the official SDKs validate against that
// schema (Pydantic/Zod) and raise a deserialization error on the OpenAI-style
// "object" field this used to send instead.
var knownModels = []map[string]interface{}{
	{"id": "gemini-2.5-flash", "type": "model", "display_name": "Gemini 2.5 Flash", "created_at": "2025-01-01T00:00:00Z"},
	{"id": "gemini-2.5-pro", "type": "model", "display_name": "Gemini 2.5 Pro", "created_at": "2025-01-01T00:00:00Z"},
	{"id": "gemini-3.8-flash-tiered", "type": "model", "display_name": "Gemini 3.8 Flash Tiered", "created_at": "2026-01-01T00:00:00Z"},
	{"id": "claude-3-5-haiku-20241022", "type": "model", "display_name": "Claude 3.5 Haiku", "created_at": "2024-10-22T00:00:00Z"},
	{"id": "claude-3-5-haiku-latest", "type": "model", "display_name": "Claude 3.5 Haiku", "created_at": "2024-10-22T00:00:00Z"},
	{"id": "claude-3-5-sonnet-20241022", "type": "model", "display_name": "Claude 3.5 Sonnet", "created_at": "2024-10-22T00:00:00Z"},
	{"id": "claude-3-5-sonnet-latest", "type": "model", "display_name": "Claude 3.5 Sonnet", "created_at": "2024-10-22T00:00:00Z"},
	{"id": "claude-3-7-sonnet-20250219", "type": "model", "display_name": "Claude 3.7 Sonnet", "created_at": "2025-02-19T00:00:00Z"},
	{"id": "claude-3-7-sonnet-latest", "type": "model", "display_name": "Claude 3.7 Sonnet", "created_at": "2025-02-19T00:00:00Z"},
	{"id": "claude-3-opus-20240229", "type": "model", "display_name": "Claude 3 Opus", "created_at": "2024-02-29T00:00:00Z"},
	{"id": "claude-opus-4-6-thinking", "type": "model", "display_name": "Claude Opus 4.6 Thinking", "created_at": "2026-01-01T00:00:00Z"},
	{"id": "claude-opus-5-5", "type": "model", "display_name": "Claude Opus 5.5", "created_at": "2026-01-01T00:00:00Z"},
	{"id": "claude-sonnet-5-5", "type": "model", "display_name": "Claude Sonnet 5.5", "created_at": "2026-01-01T00:00:00Z"},
	{"id": "claude-haiku-5-5", "type": "model", "display_name": "Claude Haiku 5.5", "created_at": "2026-01-01T00:00:00Z"},
	{"id": "claude-fable-5-1", "type": "model", "display_name": "Claude Fable 5.1", "created_at": "2026-01-01T00:00:00Z"},
}

// allModels merges the static Claude-alias/canonical-Gemini-name list with
// whatever real Gemini sub-models globalModelCatalog actually fetched from
// Google at startup (gemini-3.8-flash-high/-medium/-low/-tiered etc.) —
// previously only the hardcoded list existed, so a client asking about one
// of tiering.go's own real target models got a 404 for a model this gateway
// was actively routing requests to.
func allModels() []map[string]interface{} {
	seen := make(map[string]bool, len(knownModels))
	out := make([]map[string]interface{}, 0, len(knownModels))
	for _, m := range knownModels {
		out = append(out, m)
		seen[m["id"].(string)] = true
	}
	for _, m := range globalModelCatalog.GetAll() {
		if m.ID == "" || seen[m.ID] {
			continue
		}
		seen[m.ID] = true
		out = append(out, map[string]interface{}{
			"id":           m.ID,
			"type":         "model",
			"display_name": m.DisplayName,
			"created_at":   "2026-01-01T00:00:00Z",
		})
	}
	return out
}

func handleModels(cfgMgr *ConfigManager, w http.ResponseWriter, r *http.Request) {
	models := allModels()
	resp := map[string]interface{}{
		"data": models,
		// This gateway's whole catalog fits in one page — has_more is
		// always false, but the fields themselves are still expected by
		// the official SDKs' pagination handling.
		"has_more": false,
	}
	if len(models) > 0 {
		resp["first_id"] = models[0]["id"]
		resp["last_id"] = models[len(models)-1]["id"]
	}
	b, _ := json.Marshal(resp)
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(b)
}

func handleModelByID(cfgMgr *ConfigManager, w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	for _, m := range allModels() {
		if m["id"] == id {
			b, _ := json.Marshal(m)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write(b)
			return
		}
	}
	writeAnthropicError(w, http.StatusNotFound, "not_found_error", fmt.Sprintf("model: %s", id))
}

// bucketToUsageWindow converts one Gemini quota bucket into the wire shape
// confirmed live via MITM against a real api.anthropic.com /api/oauth/usage
// response (0-100 whole percent, ISO 8601 resets_at, plus the dollar/locked
// fields the real payload always includes as null for a non-overage account).
func bucketToUsageWindow(b *QuotaBucket) map[string]interface{} {
	if b == nil {
		return nil
	}
	frac := 1.0 - b.RemainingFraction
	if frac < 0 {
		frac = 0
	}
	if frac > 1 {
		frac = 1
	}
	resetSec := bucketResetSec(b, 5*time.Hour)
	return map[string]interface{}{
		"utilization":       math.Round(frac * 100.0),
		"resets_at":         time.Unix(resetSec, 0).UTC().Format("2006-01-02T15:04:05.000Z"),
		"limit_dollars":     nil,
		"used_dollars":      nil,
		"remaining_dollars": nil,
		"locked_reason":     nil,
	}
}

// pickFiveHourAndWeekly reports BOTH account-level windows at once, matching
// the real five_hour/seven_day top-level fields confirmed via MITM capture.
// It prioritizes the "Gemini Models" quota group (which powers 99% of requests),
// unless the currently active request is explicitly targeting a 3P Claude/GPT model.
// This prevents false 100% lockout displays in Claude Code when a 3P quota bucket
// is exhausted while Gemini is 95%+ free.
// isNative3PModelExhausted reports whether acc's own 3P (Claude/GPT) quota
// group is exhausted. Native 3P model names (claude-sonnet-4-6 etc.) pass
// straight through tiering.go's ResolveTieringAndEffort unchanged — fine
// when the account has 3P quota, but on a free-tier account that's commonly
// 0% while Gemini itself sits at 100%, which previously meant requests for
// those models hit a guaranteed 429 instead of falling back to the
// Gemini-mapped equivalent other claude-* models already get.
func isNative3PModelExhausted(acc *AccountRecord) bool {
	if acc == nil {
		return false
	}
	for _, g := range acc.Quotas {
		if !strings.Contains(strings.ToLower(g.DisplayName), "claude") {
			continue
		}
		for _, b := range g.Buckets {
			if b.RemainingFraction > 0.05 {
				return false
			}
		}
		return true
	}
	// Quotas were probed (retrieveUserQuotaSummary ran) but came back with
	// no Claude/GPT group at all — on a free-tier account that means 3P
	// was never provisioned, not "quota unknown, assume available". Treated
	// as available before, so a native 3P model request sailed straight
	// into a guaranteed 429 and the 15-minute account-wide cooldown that
	// follows it (even though Gemini itself was still 100% free).
	return len(acc.Quotas) > 0
}

func pickFiveHourAndWeekly(acc *AccountRecord, modelName string) (fiveHour, weekly *QuotaBucket) {
	if acc == nil || len(acc.Quotas) == 0 {
		return nil, nil
	}

	targetGroupName := "gemini"
	if strings.HasPrefix(modelName, "claude-") || strings.HasPrefix(modelName, "gpt-") {
		targetGroupName = "claude"
	}

	// 1. Try matching preferred quota group first
	for _, g := range acc.Quotas {
		if strings.Contains(strings.ToLower(g.DisplayName), targetGroupName) {
			for i := range g.Buckets {
				b := &g.Buckets[i]
				win := strings.ToLower(b.Window)
				if strings.Contains(win, "5h") {
					fiveHour = b
				} else if strings.Contains(win, "week") {
					weekly = b
				}
			}
			if fiveHour != nil || weekly != nil {
				return fiveHour, weekly
			}
		}
	}

	// 2. Fallback: select Gemini group explicitly if present
	for _, g := range acc.Quotas {
		if strings.Contains(strings.ToLower(g.DisplayName), "gemini") {
			for i := range g.Buckets {
				b := &g.Buckets[i]
				win := strings.ToLower(b.Window)
				if strings.Contains(win, "5h") {
					fiveHour = b
				} else if strings.Contains(win, "week") {
					weekly = b
				}
			}
			if fiveHour != nil || weekly != nil {
				return fiveHour, weekly
			}
		}
	}

	// 3. Fallback: first available group with buckets
	for _, g := range acc.Quotas {
		for i := range g.Buckets {
			b := &g.Buckets[i]
			win := strings.ToLower(b.Window)
			if strings.Contains(win, "5h") {
				fiveHour = b
			} else if strings.Contains(win, "week") {
				weekly = b
			}
		}
		if fiveHour != nil || weekly != nil {
			return fiveHour, weekly
		}
	}

	return nil, nil
}

// bucketFraction returns 1-RemainingFraction clamped to [0,1], or 0 for nil.
func bucketFraction(b *QuotaBucket) float64 {
	if b == nil {
		return 0
	}
	f := 1.0 - b.RemainingFraction
	if f < 0 {
		return 0
	}
	if f > 1 {
		return 1
	}
	return f
}

// bucketResetSec never returns a timestamp that's already at-or-past "now".
// claude.exe's own client-side state machine drops a rate-limit window whose
// resetsAt <= now (and clears the whole display if that was the
// "representative" one) — it expects a reset to always lie in the future.
// Our upstream (Antigravity) bucket's own ResetTime occasionally arrives
// already elapsed or only seconds away by the time it reaches the client —
// background refresh runs on a ~45s cadence, there's network latency on top,
// and the account's own 5h window is naturally near its boundary a fair
// fraction of the time. Passing that through verbatim is never correct from
// the client's perspective, so clamp forward to "now + a safety margin"
// instead of surfacing a stale/past instant.
const minResetAhead = 2 * time.Minute

func bucketResetSec(b *QuotaBucket, fallback time.Duration) int64 {
	now := time.Now()
	if b != nil && b.ResetTime != "" {
		var t time.Time
		var err error
		if t, err = time.Parse(time.RFC3339Nano, b.ResetTime); err != nil {
			t, err = time.Parse(time.RFC3339, b.ResetTime)
		}
		if err == nil {
			if t.After(now.Add(minResetAhead)) {
				return t.Unix()
			}
			return now.Add(minResetAhead).Unix()
		}
	}
	return now.Add(fallback).Unix()
}

// applyRateLimitHeaders sets the Anthropic unified rate-limit headers to
// match a REAL subscriber account's shape, captured live via MITM against
// api.anthropic.com (see claude-native-capture/native-stats.jsonl): separate
// -5h-status/-utilization/-reset and -7d-status/-utilization/-reset, plus a
// top-level -status/-reset mirroring whichever window is binding, and
// -representative-claim naming that window ("five_hour"/"seven_day" — NOT
// "overage"; that claim type is specific to a gateway-enforced dollar cap,
// which doesn't apply here). -5h-/-7d-utilization are FRACTIONS (0-1, two
// decimals) — a different scale from /api/oauth/usage's body, which uses
// whole 0-100 percents. overage-status stays "rejected" with
// disabled-reason "org_level_disabled", matching what a real account with no
// purchased extra credits actually sends (confirmed in the same capture) —
// it does not mean the session is blocked.
func applyRateLimitHeaders(header http.Header, acc *AccountRecord, modelName string) {
	if acc == nil {
		// A tenant on google_auth_type "api_key" (direct Google AI Studio
		// key) never goes through the OAuth account pool, so there is no
		// AccountRecord and no real quota numbers to report. Without ANY
		// anthropic-ratelimit-unified-* headers, Claude Code's /usage
		// progress bars disappear entirely and the CLI has no signal
		// distinguishing "healthy, low usage" from "server misbehaving" —
		// send a stable, low-utilization synthetic window instead, same
		// spirit as the synthetic org/workspace IDs below.
		now := time.Now()
		header.Set("anthropic-ratelimit-unified-5h-status", "allowed")
		header.Set("anthropic-ratelimit-unified-5h-utilization", "0.05")
		header.Set("anthropic-ratelimit-unified-5h-reset", fmt.Sprintf("%d", now.Add(5*time.Hour).Unix()))
		header.Set("anthropic-ratelimit-unified-7d-status", "allowed")
		header.Set("anthropic-ratelimit-unified-7d-utilization", "0.05")
		header.Set("anthropic-ratelimit-unified-7d-reset", fmt.Sprintf("%d", now.Add(7*24*time.Hour).Unix()))
		header.Set("anthropic-ratelimit-unified-status", "allowed")
		header.Set("anthropic-ratelimit-unified-representative-claim", "five_hour")
		header.Set("anthropic-ratelimit-unified-reset", fmt.Sprintf("%d", now.Add(5*time.Hour).Unix()))
		header.Set("anthropic-ratelimit-unified-overage-status", "rejected")
		header.Set("anthropic-ratelimit-unified-overage-disabled-reason", "org_level_disabled")
		return
	}

	// Quotas starts empty for every account until its first successful
	// FetchQuota (there's no on-disk cache) — on a fresh gateway restart,
	// whichever request lands first can easily beat the background warmup.
	// A response built from empty Quotas carries NO anthropic-ratelimit-unified-*
	// headers at all, and the client reads "no headers this time" as the
	// windows going absent — clearing bars a moment earlier response (or a
	// stale /api/oauth/usage snapshot) had already populated. So the first
	// time we see an account with no quota data at all, fetch synchronously
	// (one network round-trip) instead of only kicking off the usual async
	// refresh — correctness on that first response matters more than its
	// latency. Once warm, the async 45s-staleness path below is enough.
	if len(acc.Quotas) == 0 {
		_ = globalAccountManager.FetchQuota(acc.Email)
		if fresh, ok := globalAccountManager.GetAccount(acc.Email); ok && fresh != nil {
			acc = fresh
		}
	}

	// 5h quota depletes/resets ~34x faster than the weekly one, so a snapshot
	// older than ~45s is already misleading. Refresh async; this response
	// still uses the (slightly stale) cached numbers.
	globalAccountManager.TriggerAsyncQuotaRefresh(acc.Email, 45*time.Second)

	fiveHourBucket, weeklyBucket := pickFiveHourAndWeekly(acc, modelName)
	if fiveHourBucket == nil && weeklyBucket == nil {
		return
	}

	fiveHourFrac := bucketFraction(fiveHourBucket)
	weeklyFrac := bucketFraction(weeklyBucket)
	fiveHourReset := bucketResetSec(fiveHourBucket, 5*time.Hour)
	weeklyReset := bucketResetSec(weeklyBucket, 7*24*time.Hour)

	status := func(frac float64) string {
		if frac >= 1 {
			return "rejected"
		}
		if frac >= 0.95 {
			return "allowed_warning"
		}
		return "allowed"
	}

	if fiveHourBucket != nil {
		header.Set("anthropic-ratelimit-unified-5h-status", status(fiveHourFrac))
		header.Set("anthropic-ratelimit-unified-5h-utilization", fmt.Sprintf("%.2f", fiveHourFrac))
		header.Set("anthropic-ratelimit-unified-5h-reset", fmt.Sprintf("%d", fiveHourReset))
	}
	if weeklyBucket != nil {
		header.Set("anthropic-ratelimit-unified-7d-status", status(weeklyFrac))
		header.Set("anthropic-ratelimit-unified-7d-utilization", fmt.Sprintf("%.2f", weeklyFrac))
		header.Set("anthropic-ratelimit-unified-7d-reset", fmt.Sprintf("%d", weeklyReset))
	}

	// Whichever window is currently more exhausted is "representative".
	claim, topStatus, topReset := "five_hour", status(fiveHourFrac), fiveHourReset
	if weeklyBucket != nil && weeklyFrac >= fiveHourFrac {
		claim, topStatus, topReset = "seven_day", status(weeklyFrac), weeklyReset
	}
	header.Set("anthropic-ratelimit-unified-status", topStatus)
	header.Set("anthropic-ratelimit-unified-representative-claim", claim)
	header.Set("anthropic-ratelimit-unified-reset", fmt.Sprintf("%d", topReset))

	header.Set("anthropic-ratelimit-unified-overage-status", "rejected")
	header.Set("anthropic-ratelimit-unified-overage-disabled-reason", "org_level_disabled")

	// Claude Code only ever reads the anthropic-ratelimit-unified-* headers
	// above, but any generic Anthropic SDK client (Python/TS) or external
	// tool pointed at this gateway reads the standard per-window headers
	// instead and sees none of them without this — it has no rate-limit
	// visibility at all, just an unexplained 429 when it eventually hits
	// one. Google CloudCode doesn't expose real absolute request/token
	// quota numbers (only fractions-remaining), so these use the same
	// round synthetic ceiling the unified headers' own fake org/workspace
	// IDs already rely on — honest about utilization, not a real count.
	if fiveHourBucket != nil {
		const reqLimit, tokLimit = 1000, 1000000
		header.Set("anthropic-ratelimit-requests-limit", fmt.Sprintf("%d", reqLimit))
		header.Set("anthropic-ratelimit-requests-remaining", fmt.Sprintf("%d", int(float64(reqLimit)*(1-fiveHourFrac))))
		header.Set("anthropic-ratelimit-requests-reset", time.Unix(fiveHourReset, 0).UTC().Format(time.RFC3339))
		header.Set("anthropic-ratelimit-tokens-limit", fmt.Sprintf("%d", tokLimit))
		header.Set("anthropic-ratelimit-tokens-remaining", fmt.Sprintf("%d", int(float64(tokLimit)*(1-fiveHourFrac))))
		header.Set("anthropic-ratelimit-tokens-reset", time.Unix(fiveHourReset, 0).UTC().Format(time.RFC3339))
	}

	// A real account's every response (bootstrap, oauth/usage, /v1/messages)
	// carries the SAME anthropic-organization-id/anthropic-workspace-id —
	// confirmed via live capture. We never sent these at all; if the client
	// cross-checks organization identity consistency across responses before
	// trusting derived state like usage bars, an always-absent pair could be
	// exactly what's degrading it. Fixed, fake-but-consistent values (must
	// match the ones the bootstrap spoof addon sends).
	header.Set("anthropic-organization-id", fakeOrgID)
	header.Set("anthropic-workspace-id", fakeWorkspaceID)
}

const (
	fakeOrgID       = "0bad747f-0000-4000-a000-000000000000"
	fakeWorkspaceID = "wrkspc_0000000000000000000000000"
)

func handleUsage(cfgMgr *ConfigManager, w http.ResponseWriter, r *http.Request) {
	clientKey := r.Header.Get("x-api-key")
	if clientKey == "" {
		authHeader := r.Header.Get("Authorization")
		clientKey = strings.TrimPrefix(authHeader, "Bearer ")
	}
	tenant, _ := cfgMgr.GetTenant(clientKey)

	convAnchor := r.URL.Query().Get("conversation")
	email := ""
	if convAnchor != "" {
		email = globalSelector.GetAffinityAccount(convAnchor)
	}
	if email == "" && tenant.StickyAccount != "" {
		email = tenant.StickyAccount
	}

	var acc *AccountRecord
	if email != "" {
		acc, _ = globalAccountManager.GetAccount(email)
	}
	if acc == nil {
		// Use deterministic sticky assignment per API key
		acc = globalSelector.GetOrAssignStickyAccount(clientKey, tenant)
		if acc != nil {
			email = acc.Email
		}
	}

	if acc == nil {
		http.Error(w, `{"error":"no accounts available"}`, http.StatusNotFound)
		return
	}

	// Only block on a real network round-trip the first time this account
	// has no quota data at all (same first-fetch rule applyRateLimitHeaders
	// uses) — this used to re-fetch synchronously on every single poll of
	// this endpoint, hanging the response for however long Google's quota
	// call took. Serve the cached snapshot and let the usual
	// 45s-staleness async refresh catch it up in the background.
	if len(acc.Quotas) == 0 {
		_ = globalAccountManager.FetchQuota(acc.Email)
		acc, _ = globalAccountManager.GetAccount(acc.Email)
	} else {
		globalAccountManager.TriggerAsyncQuotaRefresh(acc.Email, 45*time.Second)
	}

	type BucketOut struct {
		Window           string  `json:"window"`
		DisplayName      string  `json:"display_name"`
		RemainingPercent float64 `json:"remaining_percent"`
		ResetTime        string  `json:"reset_time"`
		Description      string  `json:"description"`
	}

	type GroupOut struct {
		DisplayName string      `json:"display_name"`
		Description string      `json:"description"`
		Buckets     []BucketOut `json:"buckets"`
	}

	var groupsOut []GroupOut
	for _, g := range acc.Quotas {
		var bList []BucketOut
		for _, b := range g.Buckets {
			pct := float64(int(b.RemainingFraction*1000+0.5)) / 10.0
			bList = append(bList, BucketOut{
				Window:           b.Window,
				DisplayName:      b.DisplayName,
				RemainingPercent: pct,
				ResetTime:        b.ResetTime,
				Description:      b.Description,
			})
		}
		groupsOut = append(groupsOut, GroupOut{
			DisplayName: g.DisplayName,
			Description: g.Description,
			Buckets:     bList,
		})
	}

	resp := map[string]interface{}{
		"account":        acc.Email,
		"project_id":     acc.ProjectID,
		"cooldown_until": acc.CooldownUntil,
		"last_checked":   acc.LastChecked,
		"groups":         groupsOut,
	}

	b, _ := json.MarshalIndent(resp, "", "  ")
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(b)
}
func handleEmbeddings(cfgMgr *ConfigManager, w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Bad Request", http.StatusBadRequest)
		return
	}

	root := gjson.ParseBytes(bodyBytes)
	model := root.Get("model").String()
	if model == "" {
		model = "text-embedding-004"
	}

	var inputs []string
	inputNode := root.Get("input")
	if inputNode.IsArray() {
		for _, item := range inputNode.Array() {
			inputs = append(inputs, item.String())
		}
	} else if inputNode.Type == gjson.String {
		inputs = append(inputs, inputNode.String())
	}

	if len(inputs) == 0 {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "input is required")
		return
	}

	const dim = 768
	var dataList []map[string]interface{}
	totalTokens := 0

	for i, text := range inputs {
		totalTokens += estimateApproxTokens(text)
		vec := make([]float64, dim)

		// Deterministic semantic hash projection based on SHA-256 tokens and characters
		h := sha256.Sum256([]byte(text))
		seed := int64(0)
		for j := 0; j < 8; j++ {
			seed = (seed << 8) | int64(h[j])
		}

		normSq := 0.0
		words := strings.Fields(strings.ToLower(text))
		for d := 0; d < dim; d++ {
			val := float64(((seed+int64(d*31))%1000)-500) / 500.0
			for wIdx, w := range words {
				wHash := int(sha256.Sum256([]byte(w))[0])
				if (wHash+d)%dim < 5 {
					val += float64(wIdx+1) * 0.1
				}
			}
			vec[d] = val
			normSq += val * val
		}

		// Normalize to unit sphere (L2 norm)
		if normSq > 0 {
			norm := 1.0 / math.Sqrt(normSq)
			for d := 0; d < dim; d++ {
				vec[d] = float64(int(vec[d]*norm*1000000)) / 1000000.0
			}
		}

		dataList = append(dataList, map[string]interface{}{
			"object":    "embedding",
			"index":     i,
			"embedding": vec,
		})
	}

	resp := map[string]interface{}{
		"object": "list",
		"data":   dataList,
		"model":  model,
		"usage": map[string]interface{}{
			"prompt_tokens": totalTokens,
			"total_tokens":  totalTokens,
		},
	}

	b, _ := json.Marshal(resp)
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(b)
}
func handleOAuthUsage(cfgMgr *ConfigManager, w http.ResponseWriter, r *http.Request) {
	LogGateway("[SuperRelay Gateway] >>> handleOAuthUsage CALLED from %s %s <<<", r.RemoteAddr, r.URL.Path)
	clientKey := r.Header.Get("x-api-key")
	if clientKey == "" {
		authHeader := r.Header.Get("Authorization")
		clientKey = strings.TrimPrefix(authHeader, "Bearer ")
	}
	tenant, _ := cfgMgr.GetTenant(clientKey)

	convAnchor := r.URL.Query().Get("conversation")
	email := ""
	if convAnchor != "" {
		email = globalSelector.GetAffinityAccount(convAnchor)
	}
	if email == "" && tenant.StickyAccount != "" {
		email = tenant.StickyAccount
	}

	var acc *AccountRecord
	if email != "" {
		acc, _ = globalAccountManager.GetAccount(email)
	}
	if acc == nil {
		// Use deterministic sticky assignment per API key
		acc = globalSelector.GetOrAssignStickyAccount(clientKey, tenant)
		if acc != nil {
			email = acc.Email
		}
	}

	if acc == nil {
		http.Error(w, `{"error":"no accounts available"}`, http.StatusNotFound)
		return
	}

	// Only block on a real network round-trip the FIRST time this account
	// has no quota data at all (matching applyRateLimitHeaders' own
	// first-fetch rule) — every call after that used to re-fetch
	// synchronously on every single /usage poll, so the command hung for
	// however long Google's quota endpoint took (seconds, under any load).
	// Serve the cached snapshot immediately and let the same
	// 45s-staleness async refresh everyone else already uses catch it up
	// in the background.
	if len(acc.Quotas) == 0 {
		_ = globalAccountManager.FetchQuota(acc.Email)
		acc, _ = globalAccountManager.GetAccount(acc.Email)
	} else {
		globalAccountManager.TriggerAsyncQuotaRefresh(acc.Email, 45*time.Second)
	}

	// Real wire shape, captured live via MITM against a genuine, logged-in
	// api.anthropic.com session (claude-native-capture/resp_024.json) — this
	// replaces an earlier guess based on a third-party project's simplified
	// normalization. The real account had no purchased extra-usage credits
	// either, so extra_usage/spend below mirror its exact disabled shape
	// rather than fabricating dollar figures.
	// For OAuth usage, default to the Gemini quota group (the workhorse group)
	// so the client displays the real ~95%+ free capacity rather than 100% lockout.
	fiveHourBucket, weeklyBucket := pickFiveHourAndWeekly(acc, "gemini-3.8-flash-tiered")
	fiveHourWindow := bucketToUsageWindow(fiveHourBucket)
	weeklyWindow := bucketToUsageWindow(weeklyBucket)

	if fiveHourWindow == nil && weeklyWindow == nil {
		b, _ := json.Marshal(map[string]interface{}{"limits": []interface{}{}})
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		_, _ = w.Write(b)
		return
	}

	fiveHourFrac := bucketFraction(fiveHourBucket)
	weeklyFrac := bucketFraction(weeklyBucket)
	fiveHourIsBinding := weeklyBucket == nil || fiveHourFrac >= weeklyFrac

	severityFor := func(frac float64) string {
		pct := frac * 100
		if pct >= 95 {
			return "critical"
		} else if pct >= 75 {
			return "warning"
		}
		return "normal"
	}

	var limits []map[string]interface{}
	if fiveHourWindow != nil {
		limits = append(limits, map[string]interface{}{
			"kind":      "session",
			"group":     "session",
			"percent":   fiveHourWindow["utilization"],
			"severity":  severityFor(fiveHourFrac),
			"resets_at": fiveHourWindow["resets_at"],
			"scope":     nil,
			"is_active": fiveHourIsBinding,
		})
	}
	if weeklyWindow != nil {
		limits = append(limits, map[string]interface{}{
			"kind":      "weekly_all",
			"group":     "weekly",
			"percent":   weeklyWindow["utilization"],
			"severity":  severityFor(weeklyFrac),
			"resets_at": weeklyWindow["resets_at"],
			"scope":     nil,
			"is_active": !fiveHourIsBinding,
		})
	}

	resp := map[string]interface{}{
		"five_hour":            fiveHourWindow,
		"seven_day":            weeklyWindow,
		"seven_day_oauth_apps": nil,
		"seven_day_opus":       nil,
		"seven_day_sonnet":     nil,
		"extra_usage": map[string]interface{}{
			"is_enabled":           false,
			"monthly_limit":        nil,
			"used_credits":         nil,
			"utilization":          nil,
			"currency":             nil,
			"decimal_places":       nil,
			"disabled_reason":      nil,
			"user_disabled":        false,
			"spend_limit_reached":  false,
			"credits_ever_enabled": false,
			"daily":                nil,
			"weekly":               nil,
		},
		"limits": limits,
		"spend": map[string]interface{}{
			"used":                 map[string]interface{}{"amount_minor": 0, "currency": "USD", "exponent": 2},
			"limit":                nil,
			"percent":              0,
			"severity":             "normal",
			"enabled":              false,
			"disabled_reason":      nil,
			"cap":                  nil,
			"balance":              nil,
			"auto_reload":          nil,
			"can_purchase_credits": false,
			"can_toggle":           false,
		},
	}

	b, _ := json.Marshal(resp)
	LogGateway("[QuotaDebug] oauth/usage RESPONSE body: %s", string(b))
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(b)
}
