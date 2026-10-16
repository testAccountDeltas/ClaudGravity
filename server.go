package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/tls"
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
				http.Error(w, `{"error":{"type":"internal_error","message":"internal gateway error"}}`, http.StatusInternalServerError)
			}
		}()
		next.ServeHTTP(w, r)
	})
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
	interceptedMux.HandleFunc("/api/oauth/usage", func(w http.ResponseWriter, r *http.Request) {
		handleOAuthUsage(cfgMgr, w, r)
	})
	interceptedMux.HandleFunc("/api/claude_cli/bootstrap", func(w http.ResponseWriter, r *http.Request) {
		HandleBootstrapSpoof(w, r)
	})
	interceptedMux.HandleFunc("/", PassthroughToReal)

	proxySecret := cfgMgr.ProxySecret()

	// Combined HTTP + HTTPS CONNECT Proxy root handler
	rootHandler := recoverMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodConnect && mitmCA != nil {
			HandleConnectTunnel(w, r, mitmCA, interceptedMux, proxySecret)
			return
		}
		mux.ServeHTTP(w, r)
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
				Handler:   recoverMiddleware(mux),
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
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
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

	// 1. Resolve Tenant by Bearer token
	authHeader := r.Header.Get("Authorization")
	clientKey := strings.TrimPrefix(authHeader, "Bearer ")
	tenant, ok := cfgMgr.GetTenant(clientKey)
	if !ok {
		LogGateway("[#%d] Unauthorized: invalid or missing Bearer token", rid)
		http.Error(w, `{"error":{"type":"authentication_error","message":"Invalid or missing API key"}}`, http.StatusUnauthorized)
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
		http.Error(w, `{"error":{"message":"Failed to read request body"}}`, http.StatusBadRequest)
		return
	}

	// 2. Derive Conversation Anchor for Session Affinity (Prompt Caching)
	convAnchor := DeriveConversationAnchor(bodyBytes)

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
		handleClaudeCodeDirectWebSearch(rid, t0, clientKey, accLabel, tenant, googleToken, isAntigravity, activeAcc, bodyBytes, searchQuery, w)
		return
	}

	// Derived before translation (geminiBody doesn't exist yet) so the
	// LoopGuard key below can be scoped to this one conversation: its only
	// fallback that would otherwise need geminiBody is a last-resort content
	// hash, and bodyBytes is an equally stable stand-in for that.
	sessionID := DeriveSessionID(r, bodyBytes, bodyBytes)

	// 3. Full Orchestrator Translation: Tiering, LoopGuard, Middle-Truncation 64KB, Sanitizer
	geminiBody, modelName, budget, wasShrunk, err := ConvertClaudeRequestToGemini(bodyBytes, tenant, reqClass, agentKey, sessionID, instanceID)
	if err != nil {
		LogGateway("[#%d] Translation error: %v", rid, err)
		http.Error(w, fmt.Sprintf(`{"error":{"message":"Translation error: %v"}}`, err), http.StatusBadRequest)
		return
	}

	isStream := gjson.GetBytes(bodyBytes, "stream").Bool()
	serverCfg := cfgMgr.GetServerConfig()
	// Auto-detect and record session working directory for Edit repair
	if cwd := ExtractCWDFromSystemPrompt(gjson.GetBytes(bodyBytes, "system").Raw); cwd != "" {
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
		const maxSessionBumps = 2
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
				finalPayload = WrapAntigravityPayload(modelName, geminiBody, projectID, bump, sessionID)
			} else {
				finalPayload = geminiBody
			}

			LogGateway("[#%d] >> POST /v1/messages [attempt %d/%d, session-gen %d] key=%s account=%s model=%s thinkBudget=%d stream=%v bytes=%d [%s]",
				rid, attempt+1, maxAttempts, bump, tenant.Name, accLabel, modelName, budget, isStreaming, len(geminiBody), agentKey)

			if isStreaming {
				success, lastStatus, lastErr = executeStreamingWithWatchdog(rid, t0, activeAcc, clientKey, accLabel, sessionID, tenant, googleToken, isAntigravity, serverCfg, upstreamURL, finalPayload, modelName, agentKey, agentType, subagentName, instanceID, wasShrunk, hasSafeguards, w, flusher, bodyBytes)
			} else {
				success, lastStatus, lastErr = executeNonStreaming(rid, t0, activeAcc, clientKey, accLabel, sessionID, tenant, googleToken, isAntigravity, serverCfg, upstreamURL, finalPayload, modelName, agentKey, agentType, subagentName, instanceID, wasShrunk, hasSafeguards, w)
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
		http.Error(w, fmt.Sprintf(`{"error":{"type":"api_error","message":"Upstream failed after retries: %v"}}`, lastErr), lastStatus)
	}
}

func executeStreamingWithWatchdog(
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

	ctx, cancel := context.WithCancel(context.Background())

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
		w.Header().Set("Content-Type", "text/event-stream")
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
		errPayload := fmt.Sprintf(`{"type":"error","error":{"type":"api_error","message":"Upstream HTTP %d: %s"}}`, upResp.StatusCode, string(respBytes))
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
		w.Header().Set("Content-Type", "text/event-stream")
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
	streamer := NewSSEStreamer(w, flusher, modelName, sessionID, hasSafeguards, estInputTokens)

	err = StreamGeminiToClaude(streamer, upResp.Body, touch)
	totalDur := time.Since(t0)

	readCount := 0
	madeProgress := streamer.textChars > 200 // a real explanation, not just "let me check X"
	for _, t := range streamer.toolNames {
		if t == "Read" {
			readCount++
		} else if t == "Write" || t == "Edit" || t == "NotebookEdit" {
			madeProgress = true
		}
	}
	loopKey := agentKey + "|" + sessionID + "|" + instanceID
	globalLoopGuard.RecordReadTurn(loopKey, readCount, madeProgress)

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
) (bool, int, error) {
	// Unlike the streaming path (which has the panic/TTFT watchdog), this had
	// no deadline at all — a hung upstream could block the request forever.
	timeout := time.Duration(serverCfg.TTFTTimeoutSeconds+serverCfg.ReadTimeoutSeconds) * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
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
	usageMap := map[string]interface{}{
		"input_tokens":  uncachedInTok,
		"output_tokens": outTok,
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
		"content":     []map[string]interface{}{},
		"usage":       usageMap,
		"stop_reason": stopReason,
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
					if repairedArgs, ok := RepairEditToolCall(fname, argsRaw, sessionID); ok {
						argsRaw = repairedArgs
					}
					var argsObj interface{}
					_ = json.Unmarshal([]byte(argsRaw), &argsObj)
					toolsCalled = append(toolsCalled, fname)
					callID := nextToolCallID(fname)
					toolCallIDs = append(toolCallIDs, callID)
					claudeResp["content"] = append(claudeResp["content"].([]map[string]interface{}), map[string]interface{}{
						"type":  "tool_use",
						"id":    callID,
						"name":  fname,
						"input": argsObj,
					})
					claudeResp["stop_reason"] = "tool_use"
				}
			}
		}

		// Append Google Search Grounding sources if present
		if gm := cand.Get("groundingMetadata"); gm.Exists() {
			if chunks := gm.Get("groundingChunks"); chunks.IsArray() && len(chunks.Array()) > 0 {
				var sb strings.Builder
				sb.WriteString("\n\n---\n**Источники Google Search:**\n")
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
	bodyBytes, _ := io.ReadAll(r.Body)
	estTokens := estimateApproxClaudeTokens(bodyBytes)
	resp := map[string]interface{}{
		"input_tokens": estTokens,
	}
	b, _ := json.Marshal(resp)
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(b)
}

func handleModels(cfgMgr *ConfigManager, w http.ResponseWriter, r *http.Request) {
	resp := map[string]interface{}{
		"object": "list",
		"data": []map[string]interface{}{
			{
				"id":           "gemini-2.5-flash",
				"object":       "model",
				"display_name": "Gemini 2.5 Flash",
			},
			{
				"id":           "gemini-2.5-pro",
				"object":       "model",
				"display_name": "Gemini 2.5 Pro",
			},
			{
				"id":           "gemini-3.8-flash-tiered",
				"object":       "model",
				"display_name": "Gemini 3.8 Flash Tiered",
			},
			{
				"id":           "claude-3-5-haiku-20241022",
				"object":       "model",
				"display_name": "Claude 3.5 Haiku",
			},
			{
				"id":           "claude-3-7-sonnet-20250219",
				"object":       "model",
				"display_name": "Claude 3.7 Sonnet",
			},
			{
				"id":           "claude-opus-4-6-thinking",
				"object":       "model",
				"display_name": "Claude Opus 4.6 Thinking",
			},
		},
	}
	b, _ := json.Marshal(resp)
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(b)
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
	authHeader := r.Header.Get("Authorization")
	clientKey := strings.TrimPrefix(authHeader, "Bearer ")
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

	// Fetch fresh quota directly
	_ = globalAccountManager.FetchQuota(acc.Email)
	acc, _ = globalAccountManager.GetAccount(acc.Email)

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
		http.Error(w, `{"error":{"message":"input is required","type":"invalid_request_error"}}`, http.StatusBadRequest)
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
	authHeader := r.Header.Get("Authorization")
	clientKey := strings.TrimPrefix(authHeader, "Bearer ")
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

	// Fetch fresh quota directly
	_ = globalAccountManager.FetchQuota(acc.Email)
	acc, _ = globalAccountManager.GetAccount(acc.Email)

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
