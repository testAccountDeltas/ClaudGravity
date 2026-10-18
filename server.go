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
	"path/filepath"
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
	if bump, ok := sessionBumpMap[sessionID]; ok {
		sessionBumpSeen[sessionID] = time.Now()
		return bump
	}
	return 0
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
				// "internal_error" isn't one of Anthropic's documented error
				// types (invalid_request_error/authentication_error/
				// permission_error/not_found_error/rate_limit_error/
				// api_error/overloaded_error/request_too_large) — an SDK
				// that switches on the type string falls through to its
				// generic/unknown-error handling instead of api_error's.
				writeAnthropicError(w, http.StatusInternalServerError, "api_error", "internal gateway error")
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
		h.Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, DELETE, PUT, PATCH")
		// An audit caught that the many small CLI control-endpoint
		// handlers (settings, metrics_enabled, telemetry, bootstrap,
		// policy_limits, ...) never set anthropic-organization-id the
		// way the main /v1/messages response path does. Set it here once
		// for every response that goes through this shared middleware
		// (all interceptedMux/MITM-routed registrations) instead of
		// repeating it in each individual handler closure.
		h.Set("anthropic-organization-id", fakeOrgID)
		// A Round 15 audit caught anthropic-workspace-id was declared in
		// Access-Control-Expose-Headers below (so a browser client is
		// allowed to read it) but never actually set by this shared
		// middleware — only a handful of individual response paths set it
		// directly. Every endpoint routed through here now gets it too.
		h.Set("anthropic-workspace-id", fakeWorkspaceID)
		// An audit caught anthropic-dangerous-direct-browser-access missing
		// here — the official Anthropic SDKs send this REQUEST header
		// specifically to opt into being called directly from a browser
		// (bypassing the real API's own browser-origin block); without it
		// allow-listed, the browser's own CORS preflight rejects the
		// header before the request ever reaches this gateway.
		h.Set("Access-Control-Allow-Headers", "Content-Type, Authorization, x-api-key, anthropic-version, anthropic-beta, anthropic-dangerous-direct-browser-access, x-claude-code-request-class, x-claude-code-agent-type, x-claude-code-agent-id, x-claude-code-prompt-id, x-session-id, x-claude-code-session-id")
		// An audit noted the preflight response never set Max-Age, so a
		// browser-based client re-sent an OPTIONS preflight before every
		// single request instead of caching this permissive, static
		// policy for a while.
		h.Set("Access-Control-Max-Age", "86400")
		// An audit caught the unified-overage-* headers (set further down
		// in applyRateLimitHeaders) missing from this list — they were
		// present on the wire but invisible to browser-side JS, which can
		// only read a response header a CORS preflight explicitly exposed.
		h.Set("Access-Control-Expose-Headers", "anthropic-version, request-id, retry-after, anthropic-organization-id, anthropic-workspace-id, anthropic-ratelimit-requests-limit, anthropic-ratelimit-requests-remaining, anthropic-ratelimit-requests-reset, anthropic-ratelimit-tokens-limit, anthropic-ratelimit-tokens-remaining, anthropic-ratelimit-tokens-reset, anthropic-ratelimit-input-tokens-limit, anthropic-ratelimit-input-tokens-remaining, anthropic-ratelimit-input-tokens-reset, anthropic-ratelimit-output-tokens-limit, anthropic-ratelimit-output-tokens-remaining, anthropic-ratelimit-output-tokens-reset, anthropic-ratelimit-unified-5h-status, anthropic-ratelimit-unified-5h-utilization, anthropic-ratelimit-unified-5h-reset, anthropic-ratelimit-unified-7d-status, anthropic-ratelimit-unified-7d-utilization, anthropic-ratelimit-unified-7d-reset, anthropic-ratelimit-unified-status, anthropic-ratelimit-unified-representative-claim, anthropic-ratelimit-unified-reset, anthropic-ratelimit-unified-overage-status, anthropic-ratelimit-unified-overage-disabled-reason")
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
// setSplitTokenRateLimitHeaders sets the input/output-specific token
// rate-limit headers Anthropic's spec defines alongside (not instead of)
// the aggregated anthropic-ratelimit-tokens-* triplet this gateway already
// sends. These are synthetic, like the aggregated numbers they split — the
// gateway has no real per-tenant quota to report — using an 80/20
// input/output split as a representative approximation of a typical
// Claude-Code-shaped workload (large context reads, comparatively short
// generations).
func setSplitTokenRateLimitHeaders(header http.Header, tokLimit int, remainingFrac float64, reset time.Time) {
	inputLimit := int(float64(tokLimit) * 0.8)
	outputLimit := tokLimit - inputLimit
	resetStr := reset.UTC().Format(time.RFC3339)
	header.Set("anthropic-ratelimit-input-tokens-limit", fmt.Sprintf("%d", inputLimit))
	header.Set("anthropic-ratelimit-input-tokens-remaining", fmt.Sprintf("%d", int(float64(inputLimit)*remainingFrac)))
	header.Set("anthropic-ratelimit-input-tokens-reset", resetStr)
	header.Set("anthropic-ratelimit-output-tokens-limit", fmt.Sprintf("%d", outputLimit))
	header.Set("anthropic-ratelimit-output-tokens-remaining", fmt.Sprintf("%d", int(float64(outputLimit)*remainingFrac)))
	header.Set("anthropic-ratelimit-output-tokens-reset", resetStr)
}

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
	case http.StatusPaymentRequired:
		// Anthropic's documented error type for exceeding an account's
		// spend/billing limit — was falling through to the generic 4xx
		// default (invalid_request_error) below, giving a client no way
		// to distinguish "your request is malformed" from "you're out
		// of credit."
		return "billing_error"
	case 529:
		return "overloaded_error"
	default:
		// An audit caught that unhandled 4xx statuses (405, 415, 422, …)
		// fell into api_error, but Anthropic's spec classifies every
		// client (4xx) error as invalid_request_error — api_error is
		// reserved for the gateway's own 5xx failures.
		if status >= 400 && status < 500 {
			return "invalid_request_error"
		}
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
	changed := false
	// An audit caught that remapping to gemini-2.5-flash for the public
	// Gemini Developer API (server.go, right before this function runs)
	// only fixes the model name — translator.go may have already baked in
	// thinkingBudget: -1 (ModeAuto), which only the internal CloudCode/
	// Antigravity endpoint understands; the public API rejects it outright
	// with 400 INVALID_ARGUMENT. Pin it to a fixed, broadly-supported
	// budget instead when present.
	if genCfg, ok := root["generationConfig"].(map[string]interface{}); ok {
		if thinkCfg, ok := genCfg["thinkingConfig"].(map[string]interface{}); ok {
			if budget, ok := thinkCfg["thinkingBudget"].(float64); ok && budget < 0 {
				thinkCfg["thinkingBudget"] = 8192
				changed = true
			}
		}
	}
	contents, ok := root["contents"].([]interface{})
	if !ok {
		if changed {
			if out, err := json.Marshal(root); err == nil {
				return out
			}
		}
		return geminiBody
	}
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
			} else if _, hasSig := pm["thoughtSignature"]; hasSig {
				// A functionCall part carries its own thoughtSignature too
				// (translator.go sets one on every replayed tool_use, to
				// preserve call identity) — it has no "thought" key at all,
				// so the check above missed it, and thoughtSignature is
				// output-only on the public API the exact same way "thought"
				// itself is.
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
	// An audit caught these were only ever set on success responses
	// (applyRateLimitHeaders) — a client expecting them on every response,
	// error or not, saw them vanish specifically when something went
	// wrong.
	w.Header().Set("anthropic-organization-id", fakeOrgID)
	w.Header().Set("anthropic-workspace-id", fakeWorkspaceID)
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
		// Was "Retry-After: 5" next to a 15-minute unified-reset — an
		// audit caught that a client honoring Retry-After literally
		// (reasonable: it's the standard HTTP header) retries in 5
		// seconds and hits the exact same cooldown again immediately.
		// 900s matches the unified-reset window below.
		w.Header().Set("Retry-After", "900")
		// Claude Code's own rate-limit backoff/countdown reads
		// anthropic-ratelimit-unified-reset, not Retry-After — without it,
		// a 429 reported through this path (as opposed to
		// applyRateLimitHeaders' real quota-derived value on a normal
		// response) left the CLI with no reset time to count down to.
		// 15 minutes matches MarkAccountCooldown's own cooldown window for
		// a 429/RESOURCE_EXHAUSTED account elsewhere in this file.
		if w.Header().Get("anthropic-ratelimit-unified-reset") == "" {
			w.Header().Set("anthropic-ratelimit-unified-reset", fmt.Sprintf("%d", time.Now().Add(15*time.Minute).Unix()))
			w.Header().Set("anthropic-ratelimit-unified-status", "rejected")
		}
		// A 429 claiming full quota is still left is a contradiction a
		// careful client could reasonably act on — zero these out too.
		w.Header().Set("anthropic-ratelimit-requests-remaining", "0")
		w.Header().Set("anthropic-ratelimit-tokens-remaining", "0")
		// An audit caught that this path never sent the *-limit headers
		// at all (only *-remaining and the reset timestamps) — an SDK or
		// telemetry hook computing quota utilization as remaining/limit
		// either sees a missing key or divides by zero. Match
		// applyRateLimitHeaders' api_key-tenant defaults.
		if w.Header().Get("anthropic-ratelimit-requests-limit") == "" {
			w.Header().Set("anthropic-ratelimit-requests-limit", "1000")
		}
		if w.Header().Get("anthropic-ratelimit-tokens-limit") == "" {
			w.Header().Set("anthropic-ratelimit-tokens-limit", "1000000")
			setSplitTokenRateLimitHeaders(w.Header(), 1000000, 0, time.Now().Add(15*time.Minute))
		}
		// An audit caught that only the Claude-Code-specific
		// unified-reset (a Unix timestamp) was ever set on a 429 — a
		// generic Anthropic SDK reads the standard requests-reset/
		// tokens-reset headers instead, in RFC 3339 UTC, and without them
		// has no reset time to back off to.
		resetAt := time.Now().Add(15 * time.Minute).UTC().Format(time.RFC3339)
		if w.Header().Get("anthropic-ratelimit-requests-reset") == "" {
			w.Header().Set("anthropic-ratelimit-requests-reset", resetAt)
		}
		if w.Header().Get("anthropic-ratelimit-tokens-reset") == "" {
			w.Header().Set("anthropic-ratelimit-tokens-reset", resetAt)
		}
	} else if errType == "overloaded_error" {
		w.Header().Set("Retry-After", "3")
		// An audit caught that a 529 never set ANY
		// anthropic-ratelimit-unified-* headers — Claude Code's own
		// backoff/countdown display reads anthropic-ratelimit-unified-reset
		// regardless of which status triggered it (not just 429), and
		// without it on an overload response the CLI has nothing to count
		// down to. A short window matches the short Retry-After above —
		// Gemini overload is typically transient, unlike a real quota
		// exhaustion.
		if w.Header().Get("anthropic-ratelimit-unified-reset") == "" {
			w.Header().Set("anthropic-ratelimit-unified-reset", fmt.Sprintf("%d", time.Now().Add(30*time.Second).Unix()))
			w.Header().Set("anthropic-ratelimit-unified-status", "rejected")
		}
		// A follow-up audit caught that 529 never got the same *-limit/
		// *-remaining/*-reset headers the 429 branch above already has —
		// same reasoning applies here too (a telemetry hook computing
		// utilization, or a generic SDK reading the standard reset
		// headers instead of the Claude-Code-specific unified one), just
		// with the shorter 30s window matching this transient-overload
		// Retry-After instead of 429's 15-minute real-quota-exhaustion one.
		if w.Header().Get("anthropic-ratelimit-requests-limit") == "" {
			w.Header().Set("anthropic-ratelimit-requests-limit", "1000")
		}
		if w.Header().Get("anthropic-ratelimit-tokens-limit") == "" {
			w.Header().Set("anthropic-ratelimit-tokens-limit", "1000000")
			setSplitTokenRateLimitHeaders(w.Header(), 1000000, 0, time.Now().Add(30*time.Second))
		}
		w.Header().Set("anthropic-ratelimit-requests-remaining", "0")
		w.Header().Set("anthropic-ratelimit-tokens-remaining", "0")
		resetAt529 := time.Now().Add(30 * time.Second).UTC().Format(time.RFC3339)
		if w.Header().Get("anthropic-ratelimit-requests-reset") == "" {
			w.Header().Set("anthropic-ratelimit-requests-reset", resetAt529)
		}
		if w.Header().Get("anthropic-ratelimit-tokens-reset") == "" {
			w.Header().Set("anthropic-ratelimit-tokens-reset", resetAt529)
		}
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

	serverCfgAtStartup := cfgMgr.GetServerConfig()
	InitSpecCapture(serverCfgAtStartup.SpecCaptureIsEnabled(), serverCfgAtStartup.SpecCaptureDir, serverCfgAtStartup.SpecCaptureRetentionDays)
	rawChunksDir := filepath.Join(serverCfgAtStartup.SpecCaptureDir, "raw_chunks")
	InitSpecCaptureRawChunks(serverCfgAtStartup.SpecCaptureRawChunksIsEnabled(), rawChunksDir,
		serverCfgAtStartup.SpecCaptureRawChunksMaxSessions, serverCfgAtStartup.SpecCaptureRawChunksMaxBytes)

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
			// An audit caught that every 405 response here omitted the
			// RFC 7231-required Allow header naming which methods ARE
			// accepted — a well-behaved client has nothing to fall back
			// to besides guessing.
			w.Header().Set("Allow", "POST")
			writeAnthropicError(w, http.StatusMethodNotAllowed, "invalid_request_error", "Method Not Allowed")
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
	// An audit caught only the "claude_cli" spelling was registered — some
	// CLI versions/clients call the "claude_code" spelling instead.
	mux.HandleFunc("/api/claude_code/bootstrap", func(w http.ResponseWriter, r *http.Request) {
		HandleBootstrapSpoof(w, r)
	})
	mux.HandleFunc("/v1/embeddings", func(w http.ResponseWriter, r *http.Request) {
		handleEmbeddings(cfgMgr, w, r)
	})
	// /v1/complete (legacy text completions) and /v1/messages/batches were
	// completely unregistered — an audit caught that both fell through to
	// the dashboard's own catch-all "/" handler, which demands HTTP Basic
	// Auth and returns an HTML/text-plain 401 instead of a proper Anthropic
	// JSON error envelope. Neither is actually implemented; both just say
	// so clearly instead of looking like a broken dashboard login.
	mux.HandleFunc("/v1/complete", handleDeprecatedComplete)
	mux.HandleFunc("/v1/messages/batches", handleUnsupportedBatches)
	mux.HandleFunc("/v1/messages/batches/", handleUnsupportedBatches)
	// Same reasoning: a typo'd or unimplemented /v1/* path (e.g.
	// /v1/mesages) fell through to the dashboard's HTML/Basic-Auth catch-all
	// instead of a clean Anthropic 404 — register a prefix fallback last so
	// Go's ServeMux only reaches it when nothing more specific above matched.
	mux.HandleFunc("/v1/", handleUnknownV1Path)

	// Claude Code CLI internal control endpoints: when ANTHROPIC_BASE_URL is set,
	// the CLI queries these paths. If unhandled on mux, they fall through to "/"
	// (dashboard HTML), breaking profile validation, metrics, and remote settings.
	// An audit caught that these spoofed CLI responses all shared the
	// EXACT SAME UUID for the account, the organization, AND settings —
	// three conceptually different entities. A client that caches
	// fetched objects keyed by UUID (not by UUID+type) could overwrite
	// one entity's cached fields with another's. fakeAccountID (account/
	// settings) and fakeOrgID (organization) below are now distinct.
	handleProfileJSON := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("anthropic-organization-id", fakeOrgID)
		w.Header().Set("anthropic-workspace-id", fakeWorkspaceID)
		// An audit flagged (independent of its own unverifiable claude.exe
		// decompilation) that this response only had "email_address", not
		// "email" — checkable directly against our own structural claim
		// here. Keep both so any client reading either name works.
		_, _ = w.Write([]byte(`{"account":{"uuid":"` + fakeAccountID + `","email":"user@claudgravity.local","email_address":"user@claudgravity.local"},"organization":{"uuid":"` + fakeOrgID + `","name":"ClaudGravity Workspace"}}`))
	}
	handleValidateJSON := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// Same audit flagged this response lacked account_uuid/
		// organization_uuid alongside "valid" — add them so a client that
		// expects to read them back after validating doesn't see them as
		// missing. An audit caught these were also hardcoded to the SAME
		// value as /api/oauth/profile's nested account.uuid/organization.uuid
		// here, inconsistent as flat vs. nested keys but at least
		// referencing the SAME two distinct underlying entities now.
		_, _ = w.Write([]byte(`{"valid":true,"account_uuid":"` + fakeAccountID + `","organization_uuid":"` + fakeOrgID + `"}`))
	}
	// An audit caught these minor CLI control-endpoint handlers never set
	// anthropic-organization-id the way the main /v1/messages response
	// does — commonAPIMiddleware now sets it for the interceptedMux/MITM
	// side; each closure below sets it too for the plain mux (direct
	// HTTP) side, which goes through no shared wrapper.
	handleMetricsJSON := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("anthropic-organization-id", fakeOrgID)
		_, _ = w.Write([]byte(`{"metrics_logging_enabled":false}`))
	}
	handleSettingsJSON := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("anthropic-organization-id", fakeOrgID)
		_, _ = w.Write([]byte(`{"uuid":"` + fakeAccountID + `","checksum":"none","settings":{}}`))
	}
	handleTelemetryJSON := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("anthropic-organization-id", fakeOrgID)
		_, _ = w.Write([]byte(`{}`))
	}
	// An audit caught these CLI management/telemetry endpoints were
	// never registered at all — in direct HTTP mode they fell through to
	// the dashboard handler and got a 401 Basic Auth challenge; over
	// HTTPS_PROXY/MITM they fell through to PassthroughToReal and either
	// 401'd against the real api.anthropic.com or hung for its full 30s
	// timeout. Same spirit as handleProfileJSON/handleValidateJSON above.
	handleOrganizationsCurrentJSON := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("anthropic-organization-id", fakeOrgID)
		_, _ = w.Write([]byte(`{"uuid":"` + fakeOrgID + `","name":"ClaudGravity Workspace","organization_type":"enterprise","rate_limit_tier":"tier_4"}`))
	}
	// An audit caught that Go 1.22+'s ServeMux pattern "/api/organizations/{id}"
	// requires a non-empty segment there — a request for the bare
	// COLLECTION endpoint "GET /api/organizations" (no id at all) never
	// matches it, and had no other registered route to fall back to
	// either, so it fell through to the dashboard's 401 Basic Auth
	// challenge (direct mode) or PassthroughToReal (MITM mode).
	handleOrganizationsListJSON := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("anthropic-organization-id", fakeOrgID)
		_, _ = w.Write([]byte(`{"data":[{"uuid":"` + fakeOrgID + `","name":"ClaudGravity Workspace","organization_type":"enterprise","rate_limit_tier":"tier_4"}],"has_more":false,"first_id":"` + fakeOrgID + `","last_id":"` + fakeOrgID + `"}`))
	}
	handleAuthCurrentUserJSON := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("anthropic-organization-id", fakeOrgID)
		// An audit caught the real Anthropic endpoint also returns a
		// "memberships" array ({"organization":...,"role":"admin"}), not
		// just "organizations" — a strict client reading memberships to
		// determine org membership saw an empty/missing list and
		// concluded the user belongs to no organization at all.
		// A Round 15 audit caught the membership object itself was missing
		// its own "uuid" and "organization_uuid" fields — a strict client
		// reading those directly off the membership (not just nested under
		// "organization") saw them as empty/missing and failed its own
		// organization-membership validation.
		_, _ = w.Write([]byte(`{"uuid":"` + fakeAccountID + `","email":"user@claudgravity.local","organizations":[{"uuid":"` + fakeOrgID + `","name":"ClaudGravity Workspace"}],"memberships":[{"uuid":"` + fakeOrgID + `-member","organization_uuid":"` + fakeOrgID + `","organization":{"uuid":"` + fakeOrgID + `","name":"ClaudGravity Workspace"},"role":"admin"}]}`))
	}
	handleBillingJSON := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("anthropic-organization-id", fakeOrgID)
		_, _ = w.Write([]byte(`{"tier":"tier_4","status":"active","has_payment_method":true,"currency":"usd"}`))
	}
	// An audit caught these two were unhandled, same failure mode as the
	// organizations/billing endpoints above: Claude Code CLI calls
	// policy_limits on startup/compliance checks and prints a visible
	// terminal warning ("Couldn't load your organization's policy...")
	// when it 404s; claude_code_first_token_date is queried during
	// onboarding.
	handlePolicyLimitsJSON := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("anthropic-organization-id", fakeOrgID)
		_, _ = w.Write([]byte(`{"restrictions":{},"compliance_taints":[]}`))
	}
	handleFirstTokenDateJSON := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("anthropic-organization-id", fakeOrgID)
		_, _ = w.Write([]byte(`{"first_token_date":null}`))
	}
	// An audit flagged these as unregistered (independent of its own
	// unverifiable claude.exe decompile byte-offsets, which we don't rely
	// on — the structural fact that these paths are absent from this
	// file is checked directly): same failure mode as every other
	// unhandled CLI control endpoint above, either a 401 Basic Auth
	// locally or a PassthroughToReal leak to the real api.anthropic.com.
	handleNotificationPreferencesJSON := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("anthropic-organization-id", fakeOrgID)
		_, _ = w.Write([]byte(`{"account_id":"` + fakeAccountID + `","organization_id":"` + fakeOrgID + `","preferences":{"feature_preference":{"bogosort":{"enable_push":true}}},"push_reachability":{"has_active_channel":true,"platforms":["cli"]}}`))
	}
	handleOAuthAccountSettingsJSON := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("anthropic-organization-id", fakeOrgID)
		_, _ = w.Write([]byte(`{}`))
	}
	handleGroveNoticeViewedJSON := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("anthropic-organization-id", fakeOrgID)
		_, _ = w.Write([]byte(`{"viewed":true}`))
	}
	// Background cache-keepalive ping — CLI permanently disables this
	// background refresh for the session after 3 consecutive failures,
	// so leaving it 404ing/401ing isn't just one warning, it's a
	// permanently degraded prefix cache for the rest of the session.
	handleCacheTouchJSON := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("anthropic-organization-id", fakeOrgID)
		_, _ = w.Write([]byte(`{"status":"ok","touched":true}`))
	}
	handleModelSelectorJSON := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("anthropic-organization-id", fakeOrgID)
		_, _ = w.Write([]byte(`{}`))
	}
	handleResetRateLimitsJSON := func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("anthropic-organization-id", fakeOrgID)
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	}

	mux.HandleFunc("/api/claude_cli_profile", handleProfileJSON)
	mux.HandleFunc("/api/oauth/profile", handleProfileJSON)
	mux.HandleFunc("/api/oauth/validate", handleValidateJSON)
	mux.HandleFunc("/api/claude_code/organizations/metrics_enabled", handleMetricsJSON)
	// A Round 15 audit caught the real CLI request is parameterized by org
	// UUID — "/api/claude_code/organizations/{id}/metrics_enabled" — same
	// Go 1.22+ ServeMux exact-segment-count gap as claude_code_first_token_date above.
	mux.HandleFunc("/api/claude_code/organizations/{id}/metrics_enabled", handleMetricsJSON)
	mux.HandleFunc("/api/claude_code/settings", handleSettingsJSON)
	mux.HandleFunc("/managed/settings", handleSettingsJSON)
	mux.HandleFunc("/api/event_logging/", handleTelemetryJSON)
	// Go's http.ServeMux 301-redirects a bare "/api/event_logging" (no
	// trailing slash) to the slash-suffixed form registered above, which
	// drops the POST body on the redirect — register the bare path too,
	// same handler, instead of relying on the redirect.
	mux.HandleFunc("/api/event_logging", handleTelemetryJSON)
	mux.HandleFunc("/api/telemetry", handleTelemetryJSON)
	mux.HandleFunc("/api/telemetry/", handleTelemetryJSON)
	// An audit caught these two weren't registered at all (404ing and
	// spamming the terminal) — same telemetry-sink pattern as the paths
	// just above, reusing the same handler.
	mux.HandleFunc("/api/claude_code/telemetry", handleTelemetryJSON)
	mux.HandleFunc("/api/claude_code/crash_reports", handleTelemetryJSON)
	// A Round 15 audit caught some telemetry/crash-report clients send a
	// trailing slash (same convention "/api/telemetry/" above already
	// handles) — Go's ServeMux treats a trailing-slash path as a
	// genuinely different, non-matching pattern from the bare one.
	mux.HandleFunc("/api/claude_code/telemetry/", handleTelemetryJSON)
	mux.HandleFunc("/api/claude_code/crash_reports/", handleTelemetryJSON)
	mux.HandleFunc("/api/organizations", handleOrganizationsListJSON)
	mux.HandleFunc("/api/organizations/", handleOrganizationsListJSON)
	mux.HandleFunc("/api/organizations/current", handleOrganizationsCurrentJSON)
	mux.HandleFunc("/api/auth/current_user", handleAuthCurrentUserJSON)
	mux.HandleFunc("/api/organizations/current/billing", handleBillingJSON)
	mux.HandleFunc("/api/organizations/{id}/billing", handleBillingJSON)
	// An audit caught the bare "/api/organizations/{id}" (no "/billing"
	// suffix) was never registered at all — Claude Code queries this
	// directly after startup for org details. Unlike the already-handled
	// "/current" and "/{id}/billing" variants, this fell through
	// entirely: on the plain mux to the dashboard's Basic Auth 401, and
	// on interceptedMux to PassthroughToReal (this gateway's own local
	// key sent to the REAL api.anthropic.com, a guaranteed 401 there).
	mux.HandleFunc("/api/organizations/{id}", handleOrganizationsCurrentJSON)
	mux.HandleFunc("/api/claude_code/policy_limits", handlePolicyLimitsJSON)
	mux.HandleFunc("/api/organization/claude_code_first_token_date", handleFirstTokenDateJSON)
	// An audit caught only the singular "organization" form was ever
	// registered — some CLI versions request the plural
	// "organizations" form instead.
	mux.HandleFunc("/api/organizations/claude_code_first_token_date", handleFirstTokenDateJSON)
	// A Round 15 audit caught that the real CLI request is parameterized
	// by org UUID — "/api/organizations/{id}/claude_code_first_token_date"
	// — which Go 1.22+'s ServeMux (same pattern convention used just above
	// for {id}/billing etc.) treats as a distinct, non-matching path from
	// the bare collection route, 401ing instead of 404ing cleanly.
	mux.HandleFunc("/api/organizations/{id}/claude_code_first_token_date", handleFirstTokenDateJSON)
	mux.HandleFunc("/api/claude_code/notification/preferences", handleNotificationPreferencesJSON)
	// A Round 15 audit caught these only matched Claude Code's own
	// "/oauth/"-prefixed convention — some CLI versions instead query
	// "/api/account/settings" / "/api/account/grove_notice_viewed" with
	// no "/oauth/" segment at all, 401ing against this gateway's own
	// Basic Auth check instead of reaching either handler.
	mux.HandleFunc("/api/account/settings", handleOAuthAccountSettingsJSON)
	mux.HandleFunc("/api/account/grove_notice_viewed", handleGroveNoticeViewedJSON)
	mux.HandleFunc("/api/oauth/account/settings", handleOAuthAccountSettingsJSON)
	mux.HandleFunc("/api/oauth/account/grove_notice_viewed", handleGroveNoticeViewedJSON)
	mux.HandleFunc("/v1/messages/cache_touch", handleCacheTouchJSON)
	mux.HandleFunc("/api/organizations/{id}/model_selector/{surface}", handleModelSelectorJSON)
	// An audit caught only the surface-qualified path was registered —
	// some clients call the bare collection path with no surface segment.
	mux.HandleFunc("/api/organizations/{id}/model_selector", handleModelSelectorJSON)
	mux.HandleFunc("/api/organizations/{id}/reset_rate_limits", handleResetRateLimitsJSON)

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
	interceptedMux.Handle("/api/claude_code/bootstrap", commonAPIMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
			w.Header().Set("Allow", "POST")
			writeAnthropicError(w, http.StatusMethodNotAllowed, "invalid_request_error", "Method Not Allowed")
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
	// Same gap as /v1/messages above: an HTTPS_PROXY-only client's /v1/usage
	// call previously fell through to PassthroughToReal and hit the real
	// api.anthropic.com, which 401s with no genuine subscription behind this
	// gateway's key.
	interceptedMux.Handle("/v1/usage", commonAPIMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handleUsage(cfgMgr, w, r)
	})))
	// claude.exe fires telemetry at a hardcoded api.anthropic.com URL
	// regardless of ANTHROPIC_BASE_URL, so it only ever reaches this
	// CONNECT-tunnel path, never the direct :8900 one. With no local
	// handler it fell through to PassthroughToReal, which tried to reach
	// the real api.anthropic.com and hung for passthroughClient's full 30s
	// timeout on any network that can't reach it (isolated/firewalled
	// environments) — pure background telemetry, safe to just swallow.
	interceptedMux.Handle("/api/event_logging/", commonAPIMiddleware(http.HandlerFunc(handleTelemetryJSON)))
	// Same gap as /v1/messages and /v1/usage above: these five internal
	// control endpoints were only ever registered on the plain dashboard
	// mux, so an HTTPS_PROXY-only client fell through to PassthroughToReal
	// here and got a real 401 from api.anthropic.com for every one of
	// them (no genuine Anthropic session behind this gateway's key).
	interceptedMux.Handle("/api/claude_cli_profile", commonAPIMiddleware(http.HandlerFunc(handleProfileJSON)))
	interceptedMux.Handle("/api/oauth/profile", commonAPIMiddleware(http.HandlerFunc(handleProfileJSON)))
	interceptedMux.Handle("/api/oauth/validate", commonAPIMiddleware(http.HandlerFunc(handleValidateJSON)))
	interceptedMux.Handle("/api/claude_code/organizations/metrics_enabled", commonAPIMiddleware(http.HandlerFunc(handleMetricsJSON)))
	interceptedMux.Handle("/api/claude_code/organizations/{id}/metrics_enabled", commonAPIMiddleware(http.HandlerFunc(handleMetricsJSON)))
	interceptedMux.Handle("/api/claude_code/settings", commonAPIMiddleware(http.HandlerFunc(handleSettingsJSON)))
	interceptedMux.Handle("/managed/settings", commonAPIMiddleware(http.HandlerFunc(handleSettingsJSON)))
	// Same CLI management/telemetry endpoints added to the plain mux
	// above — claude.exe's MITM-tunneled traffic needs them too.
	interceptedMux.Handle("/api/event_logging", commonAPIMiddleware(http.HandlerFunc(handleTelemetryJSON)))
	interceptedMux.Handle("/api/telemetry", commonAPIMiddleware(http.HandlerFunc(handleTelemetryJSON)))
	interceptedMux.Handle("/api/telemetry/", commonAPIMiddleware(http.HandlerFunc(handleTelemetryJSON)))
	interceptedMux.Handle("/api/claude_code/telemetry", commonAPIMiddleware(http.HandlerFunc(handleTelemetryJSON)))
	interceptedMux.Handle("/api/claude_code/crash_reports", commonAPIMiddleware(http.HandlerFunc(handleTelemetryJSON)))
	interceptedMux.Handle("/api/claude_code/telemetry/", commonAPIMiddleware(http.HandlerFunc(handleTelemetryJSON)))
	interceptedMux.Handle("/api/claude_code/crash_reports/", commonAPIMiddleware(http.HandlerFunc(handleTelemetryJSON)))
	interceptedMux.Handle("/api/organizations", commonAPIMiddleware(http.HandlerFunc(handleOrganizationsListJSON)))
	interceptedMux.Handle("/api/organizations/", commonAPIMiddleware(http.HandlerFunc(handleOrganizationsListJSON)))
	interceptedMux.Handle("/api/organizations/current", commonAPIMiddleware(http.HandlerFunc(handleOrganizationsCurrentJSON)))
	interceptedMux.Handle("/api/auth/current_user", commonAPIMiddleware(http.HandlerFunc(handleAuthCurrentUserJSON)))
	interceptedMux.Handle("/api/organizations/current/billing", commonAPIMiddleware(http.HandlerFunc(handleBillingJSON)))
	interceptedMux.Handle("/api/organizations/{id}/billing", commonAPIMiddleware(http.HandlerFunc(handleBillingJSON)))
	// Same gaps fixed on the plain mux above (bare "/api/organizations/{id}"
	// was never registered at all; only the singular "organization" form of
	// the first-token-date endpoint existed) — the MITM-tunneled path needs
	// both too, same reasoning: without them this fell through to
	// PassthroughToReal, sending this gateway's own local key to the real
	// api.anthropic.com for a guaranteed 401.
	interceptedMux.Handle("/api/organizations/{id}", commonAPIMiddleware(http.HandlerFunc(handleOrganizationsCurrentJSON)))
	interceptedMux.Handle("/api/claude_code/policy_limits", commonAPIMiddleware(http.HandlerFunc(handlePolicyLimitsJSON)))
	interceptedMux.Handle("/api/organization/claude_code_first_token_date", commonAPIMiddleware(http.HandlerFunc(handleFirstTokenDateJSON)))
	interceptedMux.Handle("/api/organizations/claude_code_first_token_date", commonAPIMiddleware(http.HandlerFunc(handleFirstTokenDateJSON)))
	interceptedMux.Handle("/api/organizations/{id}/claude_code_first_token_date", commonAPIMiddleware(http.HandlerFunc(handleFirstTokenDateJSON)))
	interceptedMux.Handle("/api/claude_code/notification/preferences", commonAPIMiddleware(http.HandlerFunc(handleNotificationPreferencesJSON)))
	interceptedMux.Handle("/api/account/settings", commonAPIMiddleware(http.HandlerFunc(handleOAuthAccountSettingsJSON)))
	interceptedMux.Handle("/api/account/grove_notice_viewed", commonAPIMiddleware(http.HandlerFunc(handleGroveNoticeViewedJSON)))
	interceptedMux.Handle("/api/oauth/account/settings", commonAPIMiddleware(http.HandlerFunc(handleOAuthAccountSettingsJSON)))
	interceptedMux.Handle("/api/oauth/account/grove_notice_viewed", commonAPIMiddleware(http.HandlerFunc(handleGroveNoticeViewedJSON)))
	interceptedMux.Handle("/v1/messages/cache_touch", commonAPIMiddleware(http.HandlerFunc(handleCacheTouchJSON)))
	interceptedMux.Handle("/api/organizations/{id}/model_selector/{surface}", commonAPIMiddleware(http.HandlerFunc(handleModelSelectorJSON)))
	interceptedMux.Handle("/api/organizations/{id}/model_selector", commonAPIMiddleware(http.HandlerFunc(handleModelSelectorJSON)))
	interceptedMux.Handle("/api/organizations/{id}/reset_rate_limits", commonAPIMiddleware(http.HandlerFunc(handleResetRateLimitsJSON)))
	interceptedMux.Handle("/v1/embeddings", commonAPIMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handleEmbeddings(cfgMgr, w, r)
	})))
	// Same reasoning as the plain mux registrations above: without these,
	// an HTTPS_PROXY-only client hitting these paths fell through to
	// PassthroughToReal, which forwards to the real api.anthropic.com and
	// gets a genuine 401 there instead of a clear local answer.
	interceptedMux.Handle("/v1/complete", commonAPIMiddleware(http.HandlerFunc(handleDeprecatedComplete)))
	interceptedMux.Handle("/v1/messages/batches", commonAPIMiddleware(http.HandlerFunc(handleUnsupportedBatches)))
	interceptedMux.Handle("/v1/messages/batches/", commonAPIMiddleware(http.HandlerFunc(handleUnsupportedBatches)))
	// An audit caught that a "/v1/" catch-all here (added for the plain
	// mux, where there's no real Anthropic to fall back to) was a mistake
	// on the MITM mux specifically: Go's ServeMux always prefers the
	// longest matching registered pattern, so ANY "/v1/*" path this
	// gateway hasn't explicitly implemented — "/v1/skills", "/v1/sessions",
	// the MCP proxy's own "/v1/..." calls — matched this "/v1/" prefix
	// instead of falling through to "/" (PassthroughToReal), getting a
	// local 404 instead of transparently reaching the real
	// api.anthropic.com. The whole point of MITM mode is that unimplemented
	// paths pass through; only explicitly-handled "/v1/*" routes should
	// ever get a local answer here.
	interceptedMux.HandleFunc("/", PassthroughToReal)

	proxySecret := cfgMgr.ProxySecret()

	// Combined HTTP + HTTPS CONNECT Proxy root handler
	rootHandler := recoverMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodConnect {
			if mitmCA != nil {
				HandleConnectTunnel(w, r, mitmCA, interceptedMux, proxySecret, serverCfg.TLSPort)
			} else {
				http.Error(w, "CONNECT proxy not available: MITM CA initialization failed", http.StatusBadGateway)
			}
			return
		}
		apiMux.ServeHTTP(w, r)
	}))

	addr := fmt.Sprintf(":%d", serverCfg.Port)
	srv := &http.Server{
		Addr:              addr,
		Handler:           rootHandler,
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
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
				Addr:              fmt.Sprintf(":%d", tlsPort),
				Handler:           recoverMiddleware(apiMux),
				TLSConfig:         &tls.Config{Certificates: []tls.Certificate{gwCert}},
				ReadHeaderTimeout: 10 * time.Second,
				IdleTimeout:       120 * time.Second,
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

// estimateInlineDataBytes sums the byte length of every base64 "data"
// string that will actually become a Gemini inlineData part: image or
// document content blocks (at the top level of a message, or nested
// inside a tool_result's own content array) whose source.type is
// "base64". A plain-text document/content source, or an unrelated "data"
// key anywhere else in the request, never reaches inlineData and must
// not count against its 20MB limit.
func estimateInlineDataBytes(root gjson.Result) int {
	total := 0
	var walkBlocks func(blocks gjson.Result, role string)
	walkBlocks = func(blocks gjson.Result, role string) {
		if !blocks.IsArray() {
			return
		}
		blocks.ForEach(func(_, block gjson.Result) bool {
			ptype := block.Get("type").String()
			switch ptype {
			case "image", "document":
				src := block.Get("source")
				if src.Get("type").String() == "base64" {
					// An audit caught this summed EVERY base64 image/document
					// block's raw length toward the 20MB inlineData limit,
					// regardless of media type — but translator.go only ever
					// sends application/pdf and image/* (excluding SVG, which
					// it decodes as text) as actual inlineData; a base64 TEXT
					// document (CSV, code, logs — common for big data files)
					// gets decoded to a plain text part and NEVER touches
					// Gemini's real inlineData limit at all, so counting it
					// here produced false HTTP 413s on large text-only
					// requests well under any real limit. Mirror translator.go's
					// own routing condition exactly instead of assuming every
					// base64 block is binary.
					mediaType := strings.ToLower(src.Get("media_type").String())
					if mediaType == "" {
						if ptype == "document" {
							mediaType = "application/pdf"
						} else {
							mediaType = "image/png"
						}
					}
					isBinary := mediaType != "image/svg+xml" && (mediaType == "application/pdf" || strings.HasPrefix(mediaType, "image/"))
					// A follow-up audit caught that even a genuinely binary
					// image/PDF never reaches inlineData at all when it sits
					// in an ASSISTANT-role message — translator.go swaps it
					// for a short text placeholder there (Gemini's model role
					// only allows text/thought/functionCall parts), so an
					// assistant-side screenshot or PDF in history still
					// shouldn't count toward a limit it structurally can't
					// touch. Only a genuinely binary block in a USER message
					// ever becomes real inlineData bytes.
					if isBinary && role != "assistant" {
						total += len(src.Get("data").String())
					}
				}
			case "tool_result":
				c := block.Get("content")
				if c.IsArray() {
					// tool_result itself is always carried in a USER turn
					// per the Anthropic spec (the role passed in here is
					// whatever the OUTER message's role was) — pass it
					// through unchanged rather than hardcoding "user", so a
					// role-inverted tool_result (already a separate,
					// rejected-elsewhere edge case) doesn't silently start
					// counting bytes it also can't actually reach inlineData
					// with.
					walkBlocks(c, role)
				}
			}
			return true
		})
	}
	root.Get("messages").ForEach(func(_, msg gjson.Result) bool {
		walkBlocks(msg.Get("content"), msg.Get("role").String())
		return true
	})
	return total
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
		if strings.HasPrefix(strings.ToLower(authHeader), "bearer ") {
			clientKey = strings.TrimSpace(authHeader[7:])
		} else {
			clientKey = authHeader
		}
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

	// An audit caught that nothing here enforced Anthropic's documented
	// 25MB request body cap (HTTP 413 request_too_large) — an unbounded
	// io.ReadAll on a client sending a huge binary payload is a memory-
	// exhaustion risk, not just a spec gap.
	const maxRequestBodyBytes = 25 * 1024 * 1024
	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBodyBytes)
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		if err.Error() == "http: request body too large" {
			writeAnthropicError(w, http.StatusRequestEntityTooLarge, "request_too_large", "Request body exceeds the 25MB limit")
			return
		}
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "Failed to read request body")
		return
	}

	globalSpecCapture.Begin(rid)
	globalSpecCapture.RecordClaudeRequest(rid, r.Header, bodyBytes)
	defer globalSpecCapture.Finish(rid)

	// An audit caught that no upfront schema check existed here — a
	// request missing/malformed on "model", "messages", or "max_tokens"
	// fell straight into account selection and burned up to 3 real
	// upstream round-trips before eventually failing, instead of this
	// instant, free rejection.
	if !gjson.ValidBytes(bodyBytes) {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "Request body must be valid JSON")
		return
	}
	if !gjson.GetBytes(bodyBytes, "messages").IsArray() || len(gjson.GetBytes(bodyBytes, "messages").Array()) == 0 {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "\"messages\": array is required and must be non-empty")
		return
	}
	if gjson.GetBytes(bodyBytes, "model").String() == "" {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "\"model\": required field is missing")
		return
	}
	if mt := gjson.GetBytes(bodyBytes, "max_tokens"); !mt.Exists() || mt.Int() <= 0 {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "\"max_tokens\": required field must be a positive integer")
		return
	}
	// An audit caught that translator.go silently swallowed two
	// tool_choice misuses the Anthropic spec requires to be a hard 400
	// instead: tool_choice provided with no "tools" array at all (the
	// whole field is meaningless without declared tools), and
	// tool_choice:{"type":"tool","name":X} naming a tool that isn't in
	// "tools" — both previously just fell back to Gemini's own default
	// "let the model decide" with no error, silently not doing what the
	// client explicitly asked for.
	// An audit caught that this gateway's own 25MB request-body cap
	// (above) is looser than Gemini's inlineData payload limit (20MB) —
	// a request between 20-25MB passes our check but still 400s upstream
	// once translated, instead of getting a clean, fast rejection here.
	// An audit caught that summing EVERY JSON key literally named "data"
	// anywhere in the whole request tree (findPaths has no concept of
	// "is this actually base64 inlineData bound for Gemini") double-
	// counted completely unrelated things against Gemini's inlineData
	// budget: a plain-TEXT document's source.data (never becomes
	// inlineData at all — translator.go decodes it straight to a text
	// part), or any tool's own "data" argument key that happens to share
	// the name. A large text attachment or an unrelated tool call could
	// trip a false 413 that has nothing to do with the limit it's
	// supposed to enforce. Walk only the specific shape that actually
	// becomes inlineData: image/document content blocks (including ones
	// nested inside a tool_result) whose source.type is "base64".
	const maxInlineDataBytes = 20 * 1024 * 1024
	totalInlineDataBytes := estimateInlineDataBytes(gjson.ParseBytes(bodyBytes))
	if totalInlineDataBytes > maxInlineDataBytes {
		writeAnthropicError(w, http.StatusRequestEntityTooLarge, "request_too_large", "Combined size of inline image/document data exceeds the 20MB upstream limit")
		return
	}
	if tc := gjson.GetBytes(bodyBytes, "tool_choice"); tc.Exists() {
		toolsArr := gjson.GetBytes(bodyBytes, "tools")
		if !toolsArr.IsArray() || len(toolsArr.Array()) == 0 {
			writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "\"tool_choice\": provided without any \"tools\" declared")
			return
		}
		// An audit caught that an invalid tool_choice.type (anything
		// other than the spec's "auto"/"any"/"tool"/"none") was never
		// rejected here, and translator.go's own switch over this same
		// field has no default case either — an unrecognized type just
		// matched nothing, silently falling back to Gemini's own AUTO
		// instead of the hard 400 Anthropic's spec requires for a
		// malformed value.
		switch tc.Get("type").String() {
		case "auto", "any", "tool", "none":
		default:
			writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", fmt.Sprintf("\"tool_choice.type\": %q is not one of auto, any, tool, none", tc.Get("type").String()))
			return
		}
		if tc.Get("type").String() == "tool" {
			wantName := tc.Get("name").String()
			found := false
			for _, t := range toolsArr.Array() {
				if t.Get("name").String() == wantName {
					found = true
					break
				}
			}
			if !found {
				writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", fmt.Sprintf("\"tool_choice\": tool %q is not present in \"tools\"", wantName))
				return
			}
		}
	}

	// 2. Derive Conversation Anchor for Session Affinity (Prompt Caching)
	headerSessionID := r.Header.Get("x-claude-code-session-id")
	if headerSessionID == "" {
		headerSessionID = r.Header.Get("x-session-id")
	}
	convAnchor := DeriveConversationAnchor(bodyBytes, headerSessionID, instanceID)

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

	// 2.2 Fast-Path Intercept: Claude Code 1-token rate-limit probe. The
	// SDK can serialize a single-text-block message's content as either a
	// bare string or [{"type":"text","text":"quota"}] — gjson's .String()
	// on the array form returns its raw JSON, never matching "quota"
	// directly, so both shapes need checking or this probe always misses
	// and falls through to a full Gemini round-trip.
	msgsArr := gjson.GetBytes(bodyBytes, "messages").Array()
	content0 := gjson.GetBytes(bodyBytes, "messages.0.content")
	isQuotaProbeContent := content0.String() == "quota"
	if !isQuotaProbeContent && content0.IsArray() {
		blocks := content0.Array()
		isQuotaProbeContent = len(blocks) == 1 && blocks[0].Get("type").String() == "text" && blocks[0].Get("text").String() == "quota"
	}
	if len(msgsArr) == 1 &&
		isQuotaProbeContent &&
		gjson.GetBytes(bodyBytes, "max_tokens").Int() == 1 {
		acc := globalSelector.GetOrAssignStickyAccount(clientKey, tenant)
		applyRateLimitHeaders(w.Header(), acc, "gemini-3.8-flash-tiered")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"msg_probe","type":"message","role":"assistant","model":"claude-opus-5-5","content":[{"type":"text","text":"ok"}],"stop_reason":"max_tokens","usage":{"input_tokens":1,"output_tokens":1}}`))
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
	geminiBody, modelName, budget, wasShrunk, syntheticReasonTools, reverseToolNameMap, prefillText, err := ConvertClaudeRequestToGemini(bodyBytes, tenant, reqClass, agentKey, sessionID, instanceID)
	if err != nil {
		LogGateway("[#%d] Translation error: %v", rid, err)
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", fmt.Sprintf("Translation error: %v", err))
		return
	}
	globalSpecCapture.RecordGeminiRequest(rid, nil, geminiBody)
	globalSpecCapture.SetSessionMeta(rid, sessionID, agentKey)

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
		// Keyed per subagent instance, not just the terminal-wide sessionID:
		// a subagent spawned in its own git worktree legitimately has a
		// different cwd than the main thread or a sibling subagent, and
		// without instanceID whichever request recorded last silently wins
		// for everyone sharing this sessionID, corrupting Edit-repair path
		// resolution for the other one.
		cwdKey := sessionID
		if instanceID != "" {
			cwdKey = sessionID + ":" + instanceID
		}
		RecordSessionCWD(cwdKey, cwd)
	}
	// Same per-instance key RepairEditToolCall's getSessionSnapshot lookup
	// already uses (via the cwdKey passed into it at its two call sites) —
	// this was still writing under the bare sessionID, so for any subagent
	// (cwdKey != sessionID) the snapshot read could never find what this
	// just wrote, silently disabling the Edit-repair fallback for every
	// subagent.
	fileSnapKey := sessionID
	if instanceID != "" {
		fileSnapKey = sessionID + ":" + instanceID
	}
	RecordSessionFileSnapshots(fileSnapKey, bodyBytes)

	hasSafeguards := gjson.GetBytes(bodyBytes, "safeguards").Exists()
	flusher, hasFlusher := w.(http.Flusher)
	isStreaming := isStream && hasFlusher

	// 4. Pre-stream Failover Loop (up to 3 attempts with account rotation)
	maxAttempts := 3
	excludedAccounts := make(map[string]bool)
	var lastErr error
	var lastStatus int = http.StatusBadGateway
	var success bool

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
				// Was left at lastStatus's initial http.StatusBadGateway
				// default — an audit caught that the whole pool being on
				// cooldown (every account quota-exhausted) is a rate-limit
				// condition, not a gateway failure, but Claude Code treats
				// 502 as a fatal connection error and aborts the session
				// instead of running its normal 429/unified-reset countdown.
				// writeAnthropicError's own 429 branch (below, at the final
				// write) sets Retry-After/unified-reset automatically once
				// this is 429.
				lastStatus = http.StatusTooManyRequests
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
		// maxBumpsPerAttempt bounds ONE request's own retry loop (see the
		// isSessionTokenWallError branch below for why this is no longer
		// checked against the session's cumulative bump count).
		const maxBumpsPerAttempt = 3
		const maxLocationGatePasses = 2 // full passes over antigravityHosts
		hostIdx := 0
		locationGatePasses := 0
		// Per-agent-instance key for both the Antigravity sessionId AND the
		// bump counter below — computed once here instead of inline inside
		// the retry loop, since getSessionBump/setSessionBump need the same
		// key WrapAntigravityPayload uses. Keying the bump by bare sessionID
		// alone meant one subagent tripping the token wall and bumping to
		// gen 1 silently bumped every OTHER subagent and the main thread
		// sharing that same terminal session on their very next request,
		// even though their own context never came close to the wall —
		// each one got a brand new Google-side session hash (generation is
		// mixed into generateStableSessionID's hash) and lost its warm
		// prefix cache for no reason.
		antigravitySessionKey := sessionID
		if instanceID != "" {
			antigravitySessionKey = sessionID + ":" + instanceID
		} else if agentType == "auxiliary" {
			// auxiliary (title/summary generation, gemini-3.8-flash-low)
			// and the main thread both carry instanceID=="" — without this,
			// they collapsed onto the exact same antigravitySessionKey
			// (bare sessionID), sharing one Google session-cache bucket
			// between two completely different models and contexts.
			antigravitySessionKey = sessionID + ":aux"
		}
		// Start from this session's last-known-good bump instead of 0: once a
		// long (>1M-token) conversation has hit the session token-wall once,
		// every later turn would otherwise repeat the same failing bump=0
		// attempt first, burning a wasted round-trip on every single turn for
		// the rest of the conversation instead of just the first one.
		bump := getSessionBump(antigravitySessionKey)
		bumpsThisAttempt := 0
		success = false
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

			// A Round 15 audit caught that the earlier totalInlineDataBytes
			// check (above, before this retry loop) only ever sums base64
			// image/document bytes — real, but not the only way to blow
			// past Gemini's real ~20MB request-body ceiling. A long
			// conversation history's own JSON (thousands of turns, each
			// under the inline-data check's radar) plus this payload's own
			// Protobuf/JSON wrapping overhead can still push the FINAL
			// wire body over the limit even when every individual image
			// passed. Catch it here, on the actual bytes about to be sent,
			// with a clean 413 instead of letting Google's own upstream
			// reject it as a confusing 500/502.
			if len(finalPayload) > maxInlineDataBytes {
				writeAnthropicError(w, http.StatusRequestEntityTooLarge, "request_too_large", "Request body exceeds the 20MB upstream limit")
				return
			}

			LogGateway("[#%d] >> POST /v1/messages [attempt %d/%d, session-gen %d] key=%s account=%s model=%s thinkBudget=%d stream=%v bytes=%d [%s]",
				rid, attempt+1, maxAttempts, bump, tenant.Name, accLabel, modelName, budget, isStreaming, len(geminiBody), agentKey)

			if isStreaming {
				success, lastStatus, lastErr = executeStreamingWithWatchdog(r.Context(), rid, t0, activeAcc, clientKey, accLabel, sessionID, tenant, googleToken, isAntigravity, serverCfg, upstreamURL, finalPayload, modelName, agentKey, agentType, subagentName, instanceID, wasShrunk, hasSafeguards, w, flusher, bodyBytes, syntheticReasonTools, reverseToolNameMap, prefillText)
			} else {
				success, lastStatus, lastErr = executeNonStreaming(r.Context(), rid, t0, activeAcc, clientKey, accLabel, sessionID, tenant, googleToken, isAntigravity, serverCfg, upstreamURL, finalPayload, modelName, agentKey, agentType, subagentName, instanceID, wasShrunk, hasSafeguards, w, bodyBytes, syntheticReasonTools, reverseToolNameMap, prefillText)
			}

			if success || !isAntigravity {
				break
			}
			if r.Context().Err() != nil {
				break
			}
			errText := fmt.Sprint(lastErr)

			if isAntigravityLocationGateError(lastStatus, errText) || isTransientNetworkError(lastStatus, errText) {
				hostIdx++
				if hostIdx >= len(antigravityHosts) {
					hostIdx = 0
					locationGatePasses++
				}
				if locationGatePasses >= maxLocationGatePasses {
					LogGateway("[#%d] !! Location/network gate exhausted all hosts %d time(s) on %s, giving up on this account", rid, locationGatePasses, accLabel)
					break
				}
				LogGateway("[#%d] !! Location/network error on %s, retrying on next host (%s): %s", rid, accLabel, antigravityHosts[hostIdx], errText)
				continue
			}

			if isSessionTokenWallError(lastStatus, errText) {
				// bump is PERSISTED across this session's whole lifetime
				// (getSessionBump above reads back whatever the last
				// request that hit the wall saved) — capping against it
				// directly meant a long, heavily-used session that
				// organically accumulated maxSessionBumps hits over HOURS
				// of real work (each one a separate request, no looping
				// involved) permanently lost the ability to recover from
				// the wall ever again, since every later request starts
				// from that same already-maxed-out bump. Each bump is
				// cheap (loses that one prefix-cache bucket, nothing more),
				// so there's no reason to cap the session's lifetime total
				// at all — only cap bumpsThisAttempt, a local counter reset
				// every request, to bound a single request's retry loop.
				if bumpsThisAttempt >= maxBumpsPerAttempt {
					break
				}
				bumpsThisAttempt++
				bump++
				setSessionBump(antigravitySessionKey, bump)
				LogGateway("[#%d] !! Session token-wall (~1,048,576) hit on %s, bumping to a fresh session bucket (gen %d) and retrying same account", rid, accLabel, bump)
				continue
			}

			break
		}

		if success {
			// success==true here only means "the response was already
			// committed to the client, so we can't fail over to another
			// account for THIS request" — the panic watchdog can send an
			// early 200 + SSE headers and the upstream call can still go on
			// to fail (429/403) afterwards. lastStatus is the real verdict:
			// pin affinity only on an actual 200, and apply the same
			// cooldown/unstick bookkeeping a pre-headers failure gets below
			// so the NEXT request isn't routed right back to a now-known-bad
			// account just because this one couldn't retry.
			if lastStatus == http.StatusOK {
				if activeAcc != nil && convAnchor != "" {
					globalSelector.RememberAffinity(convAnchor, activeAcc.Email)
				}
			} else if activeAcc != nil {
				if lastStatus == 429 || strings.Contains(fmt.Sprint(lastErr), "RESOURCE_EXHAUSTED") {
					globalAccountManager.MarkAccountCooldown(activeAcc.Email, 15*time.Minute)
				} else if lastStatus == http.StatusForbidden {
					globalAccountManager.MarkAccountCooldown(activeAcc.Email, 60*time.Minute)
					globalSelector.UnstickAccount(activeAcc.Email)
				} else if lastStatus == http.StatusUnauthorized {
					// An audit caught this early-headers-sent branch was
					// missing the SAME 401 case the below (pre-headers)
					// branch already has — a revoked token discovered only
					// AFTER the panic watchdog already committed early SSE
					// headers to the client still left acc.AccessToken
					// cached and un-invalidated, so the next request
					// (possibly for the same conversation) reused and
					// re-failed on the same dead token instead of forcing
					// a genuine refresh.
					globalAccountManager.InvalidateAccessToken(activeAcc.Email)
				} else if lastStatus == http.StatusServiceUnavailable {
					// An audit caught 503 (Google UNAVAILABLE, typically a
					// transient overload) was excluded only for THIS
					// request's own retry attempts, same gap 429/403/401
					// already had fixed — the very next user request got
					// routed right back onto the same overloaded account
					// with no cooldown at all. Shorter than 429's 15min
					// since this is more likely a brief blip than real
					// quota exhaustion, but long enough that the immediate
					// next request or two doesn't retry it either.
					globalAccountManager.MarkAccountCooldown(activeAcc.Email, 3*time.Minute)
				}
			}
			return
		}

		// The client disconnecting (Ctrl+C, closed terminal) cancels
		// baseCtx, which is exactly why the upstream call just failed —
		// that's not a verdict on activeAcc at all, and retrying on a
		// DIFFERENT account for a request nobody is listening for anymore
		// just cycles through the rest of the pool for no reason before
		// finally giving up once maxAttempts runs out.
		if r.Context().Err() != nil {
			LogGateway("[#%d] Client disconnected (%v), aborting retry loop without touching account state", rid, r.Context().Err())
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
			} else if lastStatus == http.StatusUnauthorized {
				// An audit caught that a Google 401 (token revoked, or
				// some other auth desync) only ever excluded the account
				// from THIS request's immediate retry — the cached
				// AccessToken itself was never cleared, so every
				// subsequent request kept reusing and re-failing on the
				// same rejected token until it happened to hit its real
				// ExpiresAt. Force a genuine refresh next time instead.
				globalAccountManager.InvalidateAccessToken(activeAcc.Email)
			} else if lastStatus == http.StatusServiceUnavailable {
				// Same 503-cooldown gap as the early-headers branch above.
				globalAccountManager.MarkAccountCooldown(activeAcc.Email, 3*time.Minute)
			}
		}

		if !isAntigravity {
			break // No multiple accounts to rotate for direct API key
		}

		LogGateway("[#%d] !! Pre-stream failover: attempt %d failed (status=%d, err=%v), retrying on next account...", rid, attempt+1, lastStatus, lastErr)
		time.Sleep(100 * time.Millisecond) // brief jitter before retry
	}

	// A Round 15 audit caught that this only checked lastErr, not success —
	// every success==true return path above (see the comment on `if
	// success {` further up) has already fully written a response to w,
	// including the panic-watchdog's own "event: error" SSE frame on a
	// post-headers upstream failure, which still leaves lastErr non-nil
	// for logging purposes. Calling writeAnthropicError here on top of an
	// already-committed response caused "superfluous response.WriteHeader"
	// and appended a second, raw JSON error body into the open SSE socket,
	// breaking Anthropic SDK stream parsers. Only write the error here if
	// nothing was ever sent to the client.
	if !success && lastErr != nil {
		LogGateway("[#%d] !! All pre-stream failover attempts failed: %v", rid, lastErr)
		writeAnthropicError(w, lastStatus, anthropicErrorTypeForStatus(lastStatus), fmt.Sprintf("Upstream failed after retries: %v", lastErr))
	}
}

// isModifyingToolName reports whether a called tool name represents real
// forward progress for LoopGuard's read-loop detection — not just the
// built-in Write/Edit/NotebookEdit, but any MCP-server tool whose name
// signals a write/mutation (e.g. "mcp__filesystem__write_file"). Without
// this, an agent doing real work entirely through an MCP write tool looked
// identical to one stuck re-reading the same files, since only the three
// built-in names ever cleared the read-loop streak.
func isModifyingToolName(name string) bool {
	lower := strings.ToLower(name)
	// An audit caught this switch was case-sensitive against the exact
	// Claude Code tool names, and missing other real editing tool names
	// entirely (str_replace_editor — the Anthropic text-editor tool's own
	// name in some SDK/tool-use contexts, and "update" as used by some
	// clients). A model calling any of these in a different case (or
	// using str_replace_editor/update) never counted as progress here,
	// so LoopGuard's read-streak counter never reset after a real,
	// successful edit — it could inject a stop-reading nudge immediately
	// after the model just finished modifying the file it was reading.
	switch lower {
	case "write", "edit", "notebookedit", "str_replace_editor", "update":
		return true
	}
	if !strings.HasPrefix(lower, "mcp__") {
		return false
	}
	for _, kw := range []string{"write", "edit", "create", "delete", "update", "insert", "modify", "patch"} {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	return false
}

func hasAssistantTurns(claudeBody []byte) bool {
	msgs := gjson.GetBytes(claudeBody, "messages")
	if !msgs.IsArray() {
		return false
	}
	for _, m := range msgs.Array() {
		if m.Get("role").String() == "assistant" {
			return true
		}
	}
	return false
}

// hasCacheControlBreakpoint reports whether the client's own request
// actually asked for prompt caching (an Anthropic "cache_control":
// {"type":"ephemeral"} block somewhere in system/tools/messages) — an audit
// caught that the cache_creation_input_tokens simulation below (and in
// sse.go's FinishStream) fired unconditionally whenever cache_read was zero
// and the prompt was large, regardless of whether the request ever set a
// cache_control breakpoint at all. A plain substring check is deliberately
// used instead of walking the full JSON structure: "cache_control" (with
// its quotes, a JSON-key shape, not the bare word) is specific enough that
// a false positive is practically impossible, and this runs on every
// request. A Round 15 audit suggested also requiring "ephemeral" — free to
// add since "ephemeral" is the only type Anthropic's spec defines for this
// block, so every genuine breakpoint already contains both substrings; this
// only tightens the already-narrow theoretical false-positive case (e.g. a
// user pasting a JSON snippet that quotes the key but not its value).
func hasCacheControlBreakpoint(claudeBody []byte) bool {
	return bytes.Contains(claudeBody, []byte(`"cache_control"`)) && bytes.Contains(claudeBody, []byte(`"ephemeral"`))
}

// cacheCreationThreshold returns the minimum uncached-prompt size (tokens)
// below which the cache_creation_input_tokens simulation doesn't fire.
// An audit caught this was hardcoded to 1024 everywhere — Anthropic's own
// spec sets a higher minimum cacheable prefix, 2048 tokens, specifically
// for the Haiku model family.
func cacheCreationThreshold(claudeModelName string) int64 {
	if strings.Contains(strings.ToLower(claudeModelName), "haiku") {
		return 2048
	}
	return 1024
}

// cacheBreakpointPrefixFraction approximates how much of the request's
// uncached input actually sits BEFORE the last cache_control breakpoint
// (the part that would really get written to cache) versus after it (the
// current turn's own new content, which real Anthropic billing still
// counts as plain input_tokens).
//
// An audit caught that the cache_creation_input_tokens simulation below
// unconditionally treated 100% of uncachedInTok as cache_creation and
// zeroed input_tokens entirely once a breakpoint was merely present
// anywhere — but a breakpoint near the start of a message (the common
// case: it's set on an early, large system/tool block, with new user
// content after it) means most of the "uncached" tokens are actually this
// turn's own new content, not what's being written to cache. There's no
// real token-level breakpoint signal available (same reason the whole
// thing is a simulation), so this uses the breakpoint's byte offset within
// the request body as a proxy for its proportion of the token stream —
// coarse, but far closer than treating it as 100% either way.
func cacheBreakpointPrefixFraction(claudeBody []byte) float64 {
	idx := bytes.LastIndex(claudeBody, []byte(`"cache_control"`))
	// A Round 15 audit caught that "not found at all" (idx<0) returned 1.0
	// — the worst-case "100% is cache_creation" guess — when the honest
	// answer for "there's no breakpoint" is 0.0, no simulated cache
	// behavior at all. Currently unreachable in practice (both call sites
	// already gate on hasCacheControlBreakpoint finding one first), but
	// matches this function's own documented contract and guards a future
	// caller that doesn't gate the same way.
	if idx < 0 || len(claudeBody) == 0 {
		return 0.0
	}
	// idx==0 needs no special case: it falls through to the normal
	// fraction below like any other position, which correctly yields
	// ~0.0 (the breakpoint sits at the very start, so ~0% of the body
	// precedes it) — the original bug conflated this with "not found"
	// and returned 1.0 for both, even though they're opposite cases.
	frac := float64(idx) / float64(len(claudeBody))
	if frac < 0 {
		return 0
	}
	if frac > 1 {
		return 1
	}
	return frac
}

// wantsSingleToolUse reports whether the client set
// disable_parallel_tool_use — an audit caught that this parameter was
// completely unhandled anywhere in the codebase. translator.go adds a
// best-effort system instruction asking Gemini not to generate more than
// one tool call; this flag is also used at the actual emission points
// (sse.go's HandleFunctionCall, this file's non-streaming candidate-parts
// loop) to drop any call past the first regardless of whether Gemini
// listened, since Anthropic's own clients expect the contract enforced,
// not just requested.
func wantsSingleToolUse(claudeBody []byte) bool {
	// The official Anthropic SDKs send this nested inside tool_choice, not
	// as a top-level field — see translator.go's matching comment.
	return gjson.GetBytes(claudeBody, "disable_parallel_tool_use").Bool() || gjson.GetBytes(claudeBody, "tool_choice.disable_parallel_tool_use").Bool()
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
	reverseToolNameMap map[string]string,
	prefillText string,
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
					errPayload := fmt.Sprintf(`{"type":"error","error":{"type":"api_error","message":%q}}`, fmt.Sprintf("Upstream failed after panic: %v", res.err))
					_, _ = w.Write([]byte(fmt.Sprintf("event: error\ndata: %s\n\n", errPayload)))
					// An audit caught that the panic watchdog's error paths
					// write straight to the socket, bypassing
					// SSEStreamer.SendRaw — the one place every OTHER event
					// gets captured — so a watchdog-triggered failure left
					// no trace in speccapture's record of this request.
					globalSpecCapture.AppendClaudeEvent(rid, "error", errPayload)
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
				globalSpecCapture.AppendClaudeEvent(rid, "error", errPayload)
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

		// Headers already sent: report SSE error block. respBytes is
		// Google's raw (often JSON) error body — interpolating it into a
		// hand-built JSON string via %s let its own quotes/braces corrupt
		// the outer JSON, so the client's SDK saw a parse error instead of
		// a clean APIError.
		wMu.Lock()
		errPayloadBytes, _ := json.Marshal(map[string]interface{}{
			"type": "error",
			"error": map[string]interface{}{
				"type":    anthropicErrorTypeForStatus(upResp.StatusCode),
				"message": fmt.Sprintf("Upstream HTTP %d: %s", upResp.StatusCode, string(respBytes)),
			},
		})
		_, _ = w.Write([]byte(fmt.Sprintf("event: error\ndata: %s\n\n", errPayloadBytes)))
		globalSpecCapture.AppendClaudeEvent(rid, "error", string(errPayloadBytes))
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
	// Qualify by instanceID for the same reason RecordSessionCWD's own
	// caller does (see handleMessages): a subagent spawned in its own git
	// worktree has a genuinely different cwd than the main thread or a
	// sibling subagent sharing the same terminal sessionID, and
	// RepairEditToolCall's path resolution needs the right one, not
	// whichever request happened to record last.
	cwdKey := sessionID
	if instanceID != "" {
		cwdKey = sessionID + ":" + instanceID
	}
	// Anthropic's SDKs/telemetry expect the response's "model" field to
	// echo back exactly what the client asked for — modelName by this
	// point is the internal Gemini routing alias (e.g.
	// "gemini-3.8-flash-tiered"), not the original "claude-opus-5-5" the
	// client requested, which previously leaked straight through.
	respModel := gjson.GetBytes(claudeBodyBytes, "model").String()
	if respModel == "" {
		respModel = modelName
	}
	hasPrior := hasAssistantTurns(claudeBodyBytes)
	cachePrefixFrac := 0.0
	if hasCacheControlBreakpoint(claudeBodyBytes) {
		// An audit caught the old bool-only signal made FinishStream treat
		// 100% of the uncached input as cache_creation whenever a
		// breakpoint was merely present — see cacheBreakpointPrefixFraction's
		// doc comment for why this byte-offset-based split is closer to
		// real Anthropic billing.
		cachePrefixFrac = cacheBreakpointPrefixFraction(claudeBodyBytes)
	}
	// An audit caught that stop_sequences was translated into Gemini's
	// generationConfig.stopSequences (translator.go) but never carried
	// through to the response side at all — when the model actually
	// stopped on one, Gemini's finishReason is just "STOP" (it doesn't
	// distinguish a natural end from a stop-sequence match the way
	// Anthropic's own API does), so this always reported
	// stop_reason:"end_turn" and stop_sequence:null, leaving a client
	// that branches on stop_reason=="stop_sequence" waiting forever for
	// a continuation that was never coming.
	var stopSequences []string
	if seqs := gjson.GetBytes(claudeBodyBytes, "stop_sequences"); seqs.IsArray() {
		for _, s := range seqs.Array() {
			if s.Type == gjson.String {
				stopSequences = append(stopSequences, s.String())
			}
		}
	}
	streamer := NewSSEStreamer(rid, w, flusher, respModel, cwdKey, hasSafeguards, estInputTokens, preferRussian, syntheticReasonTools, hasPrior, cachePrefixFrac, wantsSingleToolUse(claudeBodyBytes), reverseToolNameMap, stopSequences, prefillText)

	err = StreamGeminiToClaude(streamer, upResp.Body, touch)
	if err != nil {
		wMu.Lock()
		streamer.FinishWithError(err)
		flusher.Flush()
		wMu.Unlock()
	}
	totalDur := time.Since(t0)

	readCount := 0
	madeProgress := false
	for _, sig := range streamer.toolCallSignatures {
		name, isReadEquiv, isWriteEquiv := readLoopEquivalentToolName(sig)
		if isReadEquiv {
			readCount++
		} else if isModifyingToolName(name) || isWriteEquiv {
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
	realProgress := madeProgress // genuine Write/Edit only, before the text-length credit below
	if readCount == 0 && !madeProgress {
		madeProgress = streamer.textChars > 200
	}
	loopKey := agentKey + "|" + sessionID + "|" + instanceID
	isPlanModeNow := isPlanModeRequest(claudeBodyBytes)
	// An audit caught this used to pass !isPlanModeNow as a STAND-IN for
	// hasModifyingTools — see hasModifyingToolsInRequest's doc comment
	// for why that's wrong (mismatches translator.go's own threshold,
	// computed from the real declared tools, for a read-only main-agent
	// session outside Plan Mode).
	readWindowSize, _ := readLoopLimits(agentKey, hasModifyingToolsInRequest(claudeBodyBytes))
	globalLoopGuard.RecordReadTurnWithRealProgress(loopKey, readCount, madeProgress, realProgress, readWindowSize, isPlanModeNow)

	hasIgnoreTag := strings.Contains(streamer.lastText, OscIgnoreTag)
	globalLoopGuard.RecordToolCalls(loopKey, streamer.toolCallSignatures, hasIgnoreTag)

	var hitRate float64
	if streamer.inputTokens > 0 {
		hitRate = (float64(streamer.cacheTokens) / float64(streamer.inputTokens)) * 100.0
	}

	errStr := ""
	reportedStatus := http.StatusOK
	if err != nil {
		errStr = err.Error()
		if strings.Contains(errStr, "RESOURCE_EXHAUSTED") || strings.Contains(errStr, "rate_limit_error") {
			reportedStatus = http.StatusTooManyRequests
		} else if strings.Contains(errStr, "UNAVAILABLE") || strings.Contains(errStr, "overloaded_error") {
			reportedStatus = http.StatusServiceUnavailable
		} else {
			reportedStatus = http.StatusInternalServerError
		}
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
		Status:          reportedStatus,
	})

	LogGateway("[#%d] << SSE done (%.1fs [ttft=%.1fs], in_tok=%d, out_tok=%d, cache_read=%d, tools=%v, status=%d, err=%v)",
		rid, totalDur.Seconds(), ttft.Seconds(), streamer.inputTokens, streamer.outputTokens, streamer.cacheTokens, streamer.toolNames, reportedStatus, err)

	return true, reportedStatus, err
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
	reverseToolNameMap map[string]string,
	prefillText string,
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
	globalSpecCapture.RecordGeminiResponse(rid, resp.Header, respBytes)

	if resp.StatusCode != http.StatusOK {
		LogGateway("[#%d] !! Upstream non-streaming HTTP %d: %s", rid, resp.StatusCode, string(respBytes))
		return false, resp.StatusCode, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(respBytes))
	}

	geminiJSON := gjson.ParseBytes(respBytes)
	if respNode := geminiJSON.Get("response"); respNode.Exists() {
		geminiJSON = respNode
	}
	if errNode := geminiJSON.Get("error"); errNode.Exists() {
		msg := errNode.Get("message").String()
		errCode := int(errNode.Get("code").Int())
		if errCode == 0 {
			errCode = http.StatusInternalServerError
		}
		// Was: computed a Gemini-error.status-based errType here and
		// called writeAnthropicError(w, errCode, errType, ...) before
		// returning false. An audit caught that this is the ONLY
		// `return false` in this whole function that writes to w before
		// returning — every other one (failed request creation, network
		// failure, non-200 status) leaves that to the caller.
		// handleMessages's retry loop can legitimately retry a `false`
		// return (session-token-wall bump, location-gate host rotation)
		// or fall through to its own final writeAnthropicError assuming
		// "nothing was sent yet" — both paths then called
		// w.WriteHeader/w.Write a SECOND time on top of this one,
		// producing Go's "superfluous response.WriteHeader call" and two
		// concatenated JSON bodies in the client's response. Just return
		// the error; the caller's own anthropicErrorTypeForStatus(errCode)
		// reconstructs essentially the same classification from the
		// numeric code alone (429->rate_limit_error, 400->invalid_request_error,
		// etc.) when it writes the response exactly once.
		return false, errCode, fmt.Errorf("upstream error: %s", msg)
	}
	if pf := geminiJSON.Get("promptFeedback"); pf.Exists() {
		if br := pf.Get("blockReason").String(); br != "" && br != "BLOCK_REASON_UNSPECIFIED" {
			// The streaming path (sse.go) treats this as a normal completed
			// turn with stop_reason "refusal" — a real message object, not
			// an error. This path used to send writeAnthropicError's
			// {"type":"error",...} body with HTTP 200, which the official
			// Anthropic SDKs reject outright: a 200 status means "parse
			// this as a message object," so the SDK's own response
			// validation throws before the caller ever sees the refusal
			// text. Build the same shape the streaming path produces
			// instead, so a prompt blocked before any candidate exists
			// behaves identically whether this request happened to stream
			// or not.
			blockRespModel := gjson.GetBytes(claudeBodyBytes, "model").String()
			if blockRespModel == "" {
				blockRespModel = modelName
			}
			blockResp := map[string]interface{}{
				"id":      fmt.Sprintf("msg_%d", time.Now().UnixNano()),
				"type":    "message",
				"role":    "assistant",
				"model":   blockRespModel,
				"content": []map[string]interface{}{{"type": "text", "text": "I am unable to complete this request because it was flagged by upstream safety filters: " + br}},
				"usage": map[string]interface{}{
					// An audit caught this was hardcoded to 0 rather than a
					// real estimate — a safety refusal still consumed real
					// input tokens upstream (the whole prompt had to be
					// read before Google could flag it), so reporting 0
					// understated actual usage in the client's own /usage
					// tracking.
					"input_tokens":  estimateApproxClaudeTokens(claudeBodyBytes),
					"output_tokens": 0,
					// A follow-up audit caught the cache fields were missing
					// entirely here (not just zero) — a strict typed SDK
					// expecting them always present, as every other 200
					// usage object in this codebase already does, could
					// choke on their absence.
					"cache_creation_input_tokens": 0,
					"cache_read_input_tokens":     0,
				},
				"stop_reason":   "refusal",
				"stop_sequence": nil,
				"stop_details": map[string]interface{}{
					"type":        "refusal",
					"category":    "general_harms",
					"explanation": "Content blocked by safety policy",
				},
			}
			blockJSON, _ := json.Marshal(blockResp)
			// An audit caught applyRateLimitHeaders was never called here
			// — a 200 response to /v1/messages without the full
			// anthropic-ratelimit-* header set violates the spec every
			// other 200 path already honors.
			applyRateLimitHeaders(w.Header(), activeAcc, modelName)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(blockJSON)
			globalSpecCapture.RecordClaudeResponse(rid, w.Header(), blockJSON)
			return true, http.StatusOK, nil
		}
	}
	inTok := geminiJSON.Get("usageMetadata.promptTokenCount").Int()
	if inTok == 0 {
		// An audit caught sse.go's streaming path already falls back to
		// a local estimate when Gemini's response omits usageMetadata
		// entirely (observed on some safety stops/error boundaries) —
		// this non-streaming path had no equivalent, reporting
		// input_tokens:0 and corrupting client-side token accounting.
		inTok = int64(estimateApproxClaudeTokens(claudeBodyBytes))
	}
	// An audit caught the same thoughtsTokenCount omission as sse.go's
	// streaming path — Anthropic bills thinking tokens as output tokens,
	// but this only read candidatesTokenCount, undercounting output_tokens
	// by the full size of the thinking block (commonly 80-98% on a
	// thinking-heavy turn).
	outTok := geminiJSON.Get("usageMetadata.candidatesTokenCount").Int() + geminiJSON.Get("usageMetadata.thoughtsTokenCount").Int()
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
	// Gated on the request actually having asked for caching at all — an
	// audit caught this firing on every large, cache-miss prompt
	// regardless of whether cache_control was ever set, misreporting a
	// plain request as one that paid for a (more expensive) cache write.
	// An audit caught that gating this strictly on cacheTok == 0 blocked
	// ANY cache-creation simulation on a turn that also read from an
	// existing cache — but reading an earlier cache breakpoint and
	// writing a NEW one for additional content in the same turn are not
	// mutually exclusive (e.g. a long tool-use loop where the system
	// prompt/early history is already cached, and the turn also crosses
	// a new cache_control breakpoint further into its own fresh content).
	// The real signal to gate on is whether a breakpoint is present and
	// the remaining UNCACHED amount clears the threshold, not whether
	// zero bytes were read from cache.
	var cacheCreationTok int64
	if uncachedInTok > cacheCreationThreshold(gjson.GetBytes(claudeBodyBytes, "model").String()) && hasCacheControlBreakpoint(claudeBodyBytes) {
		frac := cacheBreakpointPrefixFraction(claudeBodyBytes)
		cacheCreationTok = int64(float64(uncachedInTok) * frac)
		uncachedInTok -= cacheCreationTok
	}
	// Same per-instance CWD key as the streaming path (see
	// executeStreamingWithWatchdog) — needed below for RepairEditToolCall.
	cwdKey := sessionID
	if instanceID != "" {
		cwdKey = sessionID + ":" + instanceID
	}

	if inTok > 0 {
		if cacheTok > 0 {
			setSessionCacheFraction(sessionID, float64(cacheTok)/float64(inTok))
		} else if cacheCreationTok > 0 {
			setSessionCacheFraction(sessionID, 0.85)
		}
	}

	// A follow-up audit caught that these two fields were omitted
	// entirely from a successful 200 OK's usage object whenever they
	// were zero, instead of present with an explicit 0 — the same
	// incomplete-usage-shape issue already fixed for message_delta's
	// streaming usage and the promptFeedback-refusal usage object in
	// earlier rounds. A strict SDK client expecting the full 4-field
	// usage shape on every response saw it silently incomplete here too.
	usageMap := map[string]interface{}{
		"input_tokens":                uncachedInTok,
		"output_tokens":               outTok,
		"cache_creation_input_tokens": cacheCreationTok,
		"cache_read_input_tokens":     cacheTok,
	}

	// Previously always "end_turn" — a response that hit the output-token
	// ceiling looked identical to one that finished normally, so the client
	// had no signal to know its answer was truncated.
	stopReason := "end_turn"
	var stopDetails map[string]interface{}
	finishReason := geminiJSON.Get("candidates.0.finishReason").String()
	if finishReason == "MAX_TOKENS" {
		stopReason = "max_tokens"
	} else if finishReason == "SAFETY" || finishReason == "RECITATION" || finishReason == "BLOCKLIST" ||
		finishReason == "PROHIBITED_CONTENT" || finishReason == "SPII" {
		stopReason = "refusal"
		stopDetails = map[string]interface{}{
			"type":        "refusal",
			"category":    "general_harms",
			"explanation": "Content blocked by safety policy",
		}
	}

	// Same reasoning as the streaming path's respModel (executeStreamingWithWatchdog):
	// echo back the model the client actually requested, not the internal
	// Gemini routing alias modelName holds by this point.
	respModel := gjson.GetBytes(claudeBodyBytes, "model").String()
	if respModel == "" {
		respModel = modelName
	}
	claudeResp := map[string]interface{}{
		"id":            fmt.Sprintf("msg_%d", time.Now().UnixNano()),
		"type":          "message",
		"role":          "assistant",
		"model":         respModel,
		"content":       []map[string]interface{}{},
		"usage":         usageMap,
		"stop_reason":   stopReason,
		"stop_sequence": nil,
		// A follow-up audit caught this was omitted entirely on a normal
		// successful response instead of present as an explicit null —
		// inconsistent with "stop_sequence" right above it, which is
		// always present even when nil. A strict client expecting the
		// key to exist (even if null) on every response saw it silently
		// missing instead.
		"stop_details": stopDetails,
	}

	var toolsCalled []string
	var toolCallIDs []string
	var toolCallSignatures []string
	var textChars int
	var lastText string
	var lastTurnThoughtSig string
	cand := geminiJSON.Get("candidates.0")
	if cand.Exists() {
		parts := cand.Get("content.parts")
		if parts.IsArray() {
			// An audit caught that the thinking block below falls back to
			// lastTurnThoughtSig, which only reflects parts visited SO FAR
			// in this forward pass — but Gemini commonly attaches the
			// signature to the functionCall part that FOLLOWS a thought
			// part, not to the thought part itself. sse.go's streaming
			// path already handles this correctly by deferring the
			// thinking block's close until the next part arrives and
			// backfilling its signature from that part (see
			// HandleFunctionCall's effectiveSig logic); this non-streaming
			// path finalizes the thinking block eagerly as it's visited,
			// so it never saw a later part's signature. Pre-scan the whole
			// candidate once so a later part's signature is available
			// up front, independent of visitation order.
			var candidateThoughtSig string
			for _, p := range parts.Array() {
				if s := p.Get("thoughtSignature").String(); s != "" {
					candidateThoughtSig = s
					break
				}
			}
			for _, p := range parts.Array() {
				partSig := p.Get("thoughtSignature").String()
				if partSig != "" {
					lastTurnThoughtSig = partSig
				}
				if p.Get("thought").Bool() {
					t := p.Get("text").String()
					sig := partSig
					if sig == "" {
						sig = candidateThoughtSig
					}
					if sig == "" {
						sig = lastTurnThoughtSig
					}
					contentBlocks := claudeResp["content"].([]map[string]interface{})
					// Anthropic allows at most ONE thinking block per
					// message, and it must come first — but Gemini can
					// split one thinking chain across several adjacent
					// "thought" parts in the same candidate. sse.go's
					// streaming path already coalesces these (it only
					// opens a new content_block when the type changes);
					// this path used to create a brand new block per
					// part instead, producing multiple thinking blocks
					// that fail Pydantic/Zod schema validation outright.
					// Anthropic strictly mandates at most ONE thinking block per message,
					// and it MUST be the very first block. If a thinking block already exists,
					// concatenate to it. If it doesn't exist yet, insert it at index 0.
					var existingThinking map[string]interface{}
					for _, b := range contentBlocks {
						if b["type"] == "thinking" {
							existingThinking = b
							break
						}
					}
					if existingThinking != nil {
						if t != "" {
							existingThinking["thinking"] = existingThinking["thinking"].(string) + t
						}
						if sig != "" && (existingThinking["signature"] == "" || existingThinking["signature"] == "skip_thought_signature_validator") {
							existingThinking["signature"] = sig
						}
					} else if t != "" || sig != "" {
						if sig == "" {
							sig = "skip_thought_signature_validator"
						}
						block := map[string]interface{}{
							"type":      "thinking",
							"thinking":  t,
							"signature": sig,
						}
						claudeResp["content"] = append([]map[string]interface{}{block}, contentBlocks...)
					}
				} else if t := p.Get("text").String(); t != "" {
					if len(toolsCalled) > 0 {
						// An audit caught that nothing here stopped a
						// trailing text part (after a functionCall
						// already appeared earlier in this same
						// response) from being appended as a text block
						// AFTER the tool_use block — Anthropic's spec
						// requires tool_use to be the LAST content block
						// whenever stop_reason ends up "tool_use", and a
						// strict client reading content[length-1] as the
						// tool call breaks if a text block follows it.
						// Drop it instead, matching sse.go's streaming
						// path.
						continue
					}
					claudeResp["content"] = append(claudeResp["content"].([]map[string]interface{}), map[string]interface{}{
						"type": "text",
						"text": t,
					})
					textChars += len(t)
					// A follow-up audit caught this OVERWROTE lastText
					// instead of accumulating — Gemini can split one
					// candidate's response across multiple "text" parts,
					// and both of lastText's own consumers need the FULL
					// text: stop_sequence detection (strings.HasSuffix)
					// needs the real end of the whole response, not just
					// whatever the LAST part happened to contain, and
					// hasIgnoreTag's substring search for
					// "[ignoring loop detection]" misses it entirely if
					// the tag landed in an earlier part.
					lastText += t
				} else if fc := p.Get("functionCall"); fc.Exists() {
					if len(toolsCalled) > 0 && wantsSingleToolUse(claudeBodyBytes) {
						// Client asked for at most one tool call per turn
						// (disable_parallel_tool_use) — drop any call past
						// the first rather than handing back a second
						// tool_use block it explicitly said it can't handle.
						continue
					}
					fname := fc.Get("name").String()
					if orig, ok := reverseToolNameMap[fname]; ok {
						// translator.go sanitized this tool's name for
						// Gemini (e.g. stripped a hyphen) — restore the
						// original before it reaches Claude Code, and
						// before any lookup below keyed by that original
						// name (numericEnumFields, syntheticReasonTools).
						fname = orig
					}
					argsRaw := fc.Get("args").Raw
					if argsRaw == "" || argsRaw == "null" {
						argsRaw = "{}"
					}
					// An audit caught that this called RepairEditToolCall
					// on argsRaw BEFORE unwrapping a double-encoded JSON
					// string (sse.go's streaming path does it in the
					// opposite, correct order) — Gemini occasionally
					// emits args as a STRING containing escaped JSON
					// rather than a plain object, and RepairEditToolCall's
					// sjson/gjson field lookups (old_string/new_string/
					// file_path) find nothing at all against an opaque
					// string blob, silently skipping the repair entirely.
					// Unwrap first, matching sse.go.
					argsRaw = normalizeToolCallArgsToObject(fname, argsRaw)
					argsRaw = coerceNumericEnumArgs(fname, argsRaw)
					if repairedArgs, ok := RepairEditToolCall(fname, argsRaw, cwdKey); ok {
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
					argsRaw = stripNestedOpenDictReason(fname, argsRaw)
					var argsObj interface{}
					_ = json.Unmarshal([]byte(argsRaw), &argsObj)
					// An audit caught that the nil-check below only ever
					// caught malformed JSON — if Gemini instead emitted a
					// bare string (`args: "ls -la"`) or doubly-encoded JSON
					// (`args: "{\"cmd\":\"ls\"}"`), json.Unmarshal succeeds
					// and yields a Go string (not nil), so
					// `tool_use.input` reached the client as a JSON string
					// instead of an object, crashing Anthropic SDKs' Zod
					// validation (which requires Record<string, unknown>).
					if s, ok := argsObj.(string); ok {
						var inner interface{}
						if err := json.Unmarshal([]byte(s), &inner); err == nil {
							argsObj = inner
						}
					}
					if _, ok := argsObj.(map[string]interface{}); !ok {
						// Malformed JSON (e.g. truncated mid-argument by a
						// MAX_TOKENS cutoff) leaves argsObj nil, which
						// serializes as "input": null — Anthropic's
						// Messages API requires tool_use.input to be an
						// object. Any other non-object primitive (bare
						// string, number, bool, array) is coerced the same
						// way rather than passed through.
						//
						// An audit caught that a bare string specifically
						// (Gemini emitting args:"git status" instead of
						// args:{"command":"git status"}) silently lost the
						// model's actual argument this way instead of being
						// mapped onto the tool's one obvious primary string
						// parameter, same fix as sse.go's streaming path.
						if s, ok := argsObj.(string); ok {
							if param, ok := toolPrimaryStringParam[strings.ToLower(fname)]; ok {
								argsObj = map[string]interface{}{param: s}
							} else {
								argsObj = map[string]interface{}{}
							}
						} else {
							argsObj = map[string]interface{}{}
						}
					}
					toolsCalled = append(toolsCalled, fname)
					toolCallSignatures = append(toolCallSignatures, fmt.Sprintf("%s:%s", fname, argsRaw))
					callID := nextToolCallID(fname)
					toolCallIDs = append(toolCallIDs, callID)
					callSig := partSig
					if callSig == "" {
						callSig = lastTurnThoughtSig
					}
					setToolSignature(callID, callSig)
					claudeResp["content"] = append(claudeResp["content"].([]map[string]interface{}), map[string]interface{}{
						"type":  "tool_use",
						"id":    callID,
						"name":  fname,
						"input": argsObj,
					})
					// Same MAX_TOKENS-wins-over-tool_use priority as sse.go's
					// FinishStream: a functionCall part being present
					// doesn't mean it arrived complete.
					if claudeResp["stop_reason"] != "max_tokens" && claudeResp["stop_reason"] != "refusal" {
						claudeResp["stop_reason"] = "tool_use"
					}
				}
			}
		}

		// Append Google Search Grounding sources if present. Anthropic
		// requires tool_use to be the LAST content block whenever
		// stop_reason is "tool_use" — appending this text block after one
		// would violate that on the rare response carrying both a grounding
		// citation list and a separate custom tool call.
		if gm := cand.Get("groundingMetadata"); claudeResp["stop_reason"] != "tool_use" && gm.Exists() {
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

	// Same recovery as sse.go's streaming FinishStream — Gemini's
	// finishReason for "stopped because it hit a client-requested
	// stop_sequence" is identical to a normal end ("STOP"), so without
	// this stop_reason stayed "end_turn" and stop_sequence stayed nil
	// even when the model's own output clearly ends with one of the
	// requested sequences.
	if claudeResp["stop_reason"] == "end_turn" {
		if seqs := gjson.GetBytes(claudeBodyBytes, "stop_sequences"); seqs.IsArray() {
			for _, seqResult := range seqs.Array() {
				seq := seqResult.String()
				if seq != "" && strings.HasSuffix(lastText, seq) {
					claudeResp["stop_reason"] = "stop_sequence"
					claudeResp["stop_sequence"] = seq
					break
				}
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

	// LoopGuard previously only ran in the streaming path — a client calling
	// non-streaming entirely evaded read-loop/oscillation detection no
	// matter how many consecutive unproductive Read turns it made.
	readCount := 0
	madeProgress := false
	for _, sig := range toolCallSignatures {
		name, isReadEquiv, isWriteEquiv := readLoopEquivalentToolName(sig)
		if isReadEquiv {
			readCount++
		} else if isModifyingToolName(name) || isWriteEquiv {
			madeProgress = true
		}
	}
	realProgress := madeProgress // genuine Write/Edit only, before the text-length credit below
	if readCount == 0 && !madeProgress {
		madeProgress = textChars > 200
	}
	loopKey := agentKey + "|" + sessionID + "|" + instanceID
	isPlanModeNow := isPlanModeRequest(claudeBodyBytes)
	// An audit caught this used to pass !isPlanModeNow as a STAND-IN for
	// hasModifyingTools — see hasModifyingToolsInRequest's doc comment
	// for why that's wrong (mismatches translator.go's own threshold,
	// computed from the real declared tools, for a read-only main-agent
	// session outside Plan Mode).
	readWindowSize, _ := readLoopLimits(agentKey, hasModifyingToolsInRequest(claudeBodyBytes))
	globalLoopGuard.RecordReadTurnWithRealProgress(loopKey, readCount, madeProgress, realProgress, readWindowSize, isPlanModeNow)
	hasIgnoreTag := strings.Contains(lastText, OscIgnoreTag)
	globalLoopGuard.RecordToolCalls(loopKey, toolCallSignatures, hasIgnoreTag)

	// Anthropic's Message.content has minItems:1 — a candidate with zero
	// parts at all (safety block, or just genuinely empty) otherwise left
	// this as [], same issue already guarded against in sse.go's streaming
	// path.
	if contentBlocks, ok := claudeResp["content"].([]map[string]interface{}); ok && len(contentBlocks) == 0 {
		// An audit caught that a safety-blocked candidate with genuinely
		// zero parts (finishReason SAFETY/RECITATION/etc., stopReason
		// already set to "refusal" above) fell into this same generic
		// fallback and got the literal placeholder "(empty response)"
		// instead of refusal-shaped text — sse.go's streaming path
		// already uses real refusal text for the equivalent case.
		emptyText := "(empty response)"
		if claudeResp["stop_reason"] == "refusal" {
			emptyText = "I'm not able to help with that request."
		}
		claudeResp["content"] = append(contentBlocks, map[string]interface{}{
			"type": "text",
			"text": emptyText,
		})
	}

	// An audit caught that Anthropic's assistant-prefill contract (the
	// final returned message must BEGIN with whatever text the client
	// supplied as the trailing assistant turn, with the model continuing
	// from there) was never honored in the non-streaming path — Gemini
	// has no real prefill mechanism, so sanitizer.go instructs it via a
	// system note to continue directly from the prefix WITHOUT repeating
	// it, but that prefix itself was never prepended back onto the
	// response. A client expecting content[0].text to start with its own
	// prefill (e.g. constraining JSON output with a literal "{") got
	// Gemini's completion alone and failed to parse it.
	if prefillText != "" {
		// A Round 15 audit caught this only ever looked at contentBlocks[0]
		// — when extended thinking is on, that index is the "thinking"
		// block and the real "text" block sits at index 1+, so the prefill
		// silently never got re-attached whenever thinking was enabled.
		if contentBlocks, ok := claudeResp["content"].([]map[string]interface{}); ok {
			for _, b := range contentBlocks {
				if t, ok := b["type"].(string); ok && t == "text" {
					if existing, ok := b["text"].(string); ok && !strings.HasPrefix(existing, prefillText) {
						b["text"] = prefillText + existing
					}
					break
				}
			}
		}
	}

	outJSON, _ := json.Marshal(claudeResp)
	applyRateLimitHeaders(w.Header(), activeAcc, modelName)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(outJSON)
	globalSpecCapture.RecordClaudeResponse(rid, w.Header(), outJSON)

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
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "POST")
		writeAnthropicError(w, http.StatusMethodNotAllowed, "invalid_request_error", "This endpoint requires a POST request")
		return
	}
	// Was a direct strings.TrimPrefix(..., "Bearer ") here — case-sensitive,
	// so a client sending lowercase "bearer <token>" (several HTTP
	// libraries normalize the scheme to lowercase) got the whole header
	// value treated as the key and a false 401. authenticatedTenantFromRequest
	// (added for handleModels/handleModelByID) already does this correctly.
	if _, ok := authenticatedTenantFromRequest(cfgMgr, r); !ok {
		writeAnthropicError(w, http.StatusUnauthorized, "authentication_error", "Invalid or missing API key")
		return
	}
	bodyBytes, _ := io.ReadAll(r.Body)
	// estimateApproxClaudeTokens silently skips a non-array "messages"
	// (its IsArray() guard just no-ops), so checking only Exists() here let
	// e.g. {"messages": "a string"} through as 200 OK with a tiny token
	// count instead of the 400 invalid_request_error Anthropic's spec
	// requires for a malformed messages field.
	// An audit caught this was more permissive than /v1/messages' own
	// validation: Exists() accepts {"model": ""} (an empty string field
	// still "exists"), and nothing checked messages was non-empty — an
	// empty array passes IsArray() too. /v1/messages requires both a
	// non-empty model and non-empty messages as a hard 400.
	messagesArr := gjson.GetBytes(bodyBytes, "messages")
	if !gjson.ValidBytes(bodyBytes) || !messagesArr.IsArray() || len(messagesArr.Array()) == 0 || gjson.GetBytes(bodyBytes, "model").String() == "" {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "Request body must be valid JSON with a non-empty \"model\" and non-empty \"messages\"")
		return
	}
	estTokens := estimateApproxClaudeTokens(bodyBytes)
	resp := map[string]interface{}{
		"input_tokens": estTokens,
	}
	b, _ := json.Marshal(resp)
	// An audit caught this endpoint never called applyRateLimitHeaders
	// at all (no anthropic-ratelimit-* headers, no organization/workspace
	// ID) — every OTHER successful response sets these, and client-side
	// quota-tracking tooling that expects them on every response broke
	// specifically on count_tokens. No AccountRecord was ever selected
	// for this endpoint (it's a pure local estimate), so pass nil — the
	// function already has a synthetic-but-stable fallback for exactly
	// that case.
	applyRateLimitHeaders(w.Header(), nil, gjson.GetBytes(bodyBytes, "model").String())
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
	{"id": "gemini-2.5-flash", "type": "model", "display_name": "Gemini 2.5 Flash", "created_at": "2025-01-01T00:00:00Z", "max_input_tokens": 1048576, "max_tokens": 64000, "capabilities": map[string]interface{}{"tools": true, "thinking": true, "vision": true}},
	{"id": "gemini-2.5-pro", "type": "model", "display_name": "Gemini 2.5 Pro", "created_at": "2025-01-01T00:00:00Z", "max_input_tokens": 2097152, "max_tokens": 64000, "capabilities": map[string]interface{}{"tools": true, "thinking": true, "vision": true}},
	{"id": "gemini-3.8-flash-tiered", "type": "model", "display_name": "Gemini 3.8 Flash Tiered", "created_at": "2026-01-01T00:00:00Z", "max_input_tokens": 1048576, "max_tokens": 64000, "capabilities": map[string]interface{}{"tools": true, "thinking": true, "vision": true}},
	{"id": "claude-3-5-haiku-20241022", "type": "model", "display_name": "Claude 3.5 Haiku", "created_at": "2024-10-22T00:00:00Z", "max_input_tokens": 200000, "max_tokens": 8192, "capabilities": map[string]interface{}{"tools": true, "vision": true}},
	{"id": "claude-3-5-haiku-latest", "type": "model", "display_name": "Claude 3.5 Haiku", "created_at": "2024-10-22T00:00:00Z", "max_input_tokens": 200000, "max_tokens": 8192, "capabilities": map[string]interface{}{"tools": true, "vision": true}},
	{"id": "claude-3-5-sonnet-20241022", "type": "model", "display_name": "Claude 3.5 Sonnet", "created_at": "2024-10-22T00:00:00Z", "max_input_tokens": 200000, "max_tokens": 8192, "capabilities": map[string]interface{}{"tools": true, "vision": true}},
	{"id": "claude-3-5-sonnet-latest", "type": "model", "display_name": "Claude 3.5 Sonnet", "created_at": "2024-10-22T00:00:00Z", "max_input_tokens": 200000, "max_tokens": 8192, "capabilities": map[string]interface{}{"tools": true, "vision": true}},
	{"id": "claude-3-7-sonnet-20250219", "type": "model", "display_name": "Claude 3.7 Sonnet", "created_at": "2025-02-19T00:00:00Z", "max_input_tokens": 200000, "max_tokens": 64000, "capabilities": map[string]interface{}{"tools": true, "thinking": true, "vision": true}},
	{"id": "claude-3-7-sonnet-latest", "type": "model", "display_name": "Claude 3.7 Sonnet", "created_at": "2025-02-19T00:00:00Z", "max_input_tokens": 200000, "max_tokens": 64000, "capabilities": map[string]interface{}{"tools": true, "thinking": true, "vision": true}},
	{"id": "claude-3-opus-20240229", "type": "model", "display_name": "Claude 3 Opus", "created_at": "2024-02-29T00:00:00Z", "max_input_tokens": 200000, "max_tokens": 4096, "capabilities": map[string]interface{}{"tools": true, "vision": true}},
	{"id": "claude-3-opus-latest", "type": "model", "display_name": "Claude 3 Opus", "created_at": "2024-02-29T00:00:00Z", "max_input_tokens": 200000, "max_tokens": 4096, "capabilities": map[string]interface{}{"tools": true, "vision": true}},
	{"id": "claude-opus-4-6-thinking", "type": "model", "display_name": "Claude Opus 4.6 Thinking", "created_at": "2026-01-01T00:00:00Z", "max_input_tokens": 1048576, "max_tokens": 64000, "capabilities": map[string]interface{}{"tools": true, "thinking": true, "vision": true}},
	// An audit caught these canonical model IDs were never added to the
	// catalog — GET /v1/models/claude-opus-4-5 etc. 404ed even though
	// tiering.go's own model-string matching (HasPrefix(model,
	// "claude-opus")/"claude-sonnet"/"claude-haiku") routes requests FOR
	// these exact models just fine; the 404 only ever broke SDK model
	// lookups, not actual message requests.
	{"id": "claude-opus-4-5", "type": "model", "display_name": "Claude Opus 4.5", "created_at": "2025-11-01T00:00:00Z", "max_input_tokens": 200000, "max_tokens": 64000, "capabilities": map[string]interface{}{"tools": true, "thinking": true, "vision": true}},
	{"id": "claude-opus-4-7", "type": "model", "display_name": "Claude Opus 4.7", "created_at": "2025-12-01T00:00:00Z", "max_input_tokens": 1048576, "max_tokens": 64000, "capabilities": map[string]interface{}{"tools": true, "thinking": true, "vision": true}},
	{"id": "claude-opus-4-8", "type": "model", "display_name": "Claude Opus 4.8", "created_at": "2026-01-01T00:00:00Z", "max_input_tokens": 1048576, "max_tokens": 64000, "capabilities": map[string]interface{}{"tools": true, "thinking": true, "vision": true}},
	{"id": "claude-sonnet-4-5", "type": "model", "display_name": "Claude Sonnet 4.5", "created_at": "2025-09-29T00:00:00Z", "max_input_tokens": 1048576, "max_tokens": 64000, "capabilities": map[string]interface{}{"tools": true, "thinking": true, "vision": true}},
	{"id": "claude-haiku-4-5", "type": "model", "display_name": "Claude Haiku 4.5", "created_at": "2025-10-01T00:00:00Z", "max_input_tokens": 200000, "max_tokens": 64000, "capabilities": map[string]interface{}{"tools": true, "vision": true}},
	{"id": "claude-sonnet-4-6", "type": "model", "display_name": "Claude Sonnet 4.6", "created_at": "2026-01-01T00:00:00Z", "max_input_tokens": 1048576, "max_tokens": 64000, "capabilities": map[string]interface{}{"tools": true, "thinking": true, "vision": true}},
	{"id": "claude-opus-5-5", "type": "model", "display_name": "Claude Opus 5.5", "created_at": "2026-01-01T00:00:00Z", "max_input_tokens": 1048576, "max_tokens": 128000, "capabilities": map[string]interface{}{"tools": true, "thinking": true, "vision": true}},
	{"id": "claude-sonnet-5-5", "type": "model", "display_name": "Claude Sonnet 5.5", "created_at": "2026-01-01T00:00:00Z", "max_input_tokens": 1048576, "max_tokens": 128000, "capabilities": map[string]interface{}{"tools": true, "thinking": true, "vision": true}},
	{"id": "claude-haiku-5-5", "type": "model", "display_name": "Claude Haiku 5.5", "created_at": "2026-01-01T00:00:00Z", "max_input_tokens": 1048576, "max_tokens": 64000, "capabilities": map[string]interface{}{"tools": true, "thinking": true, "vision": true}},
	{"id": "claude-fable-5-1", "type": "model", "display_name": "Claude Fable 5.1", "created_at": "2026-01-01T00:00:00Z", "max_input_tokens": 1048576, "max_tokens": 128000, "capabilities": map[string]interface{}{"tools": true, "thinking": true, "vision": true}},
	// An audit caught that Claude Code's Agent/Task tool can request a
	// subagent model by its bare alias ("haiku", "sonnet", "opus",
	// "fable", "mythos" — no "claude-" prefix; tiering.go already maps
	// these to a real Gemini target, see ResolveTieringAndEffort), but
	// GET /v1/models/{alias} only ever matched a full canonical id,
	// 404ing on the exact short names Claude Code itself sends.
	{"id": "haiku", "type": "model", "display_name": "Claude Haiku (alias)", "created_at": "2026-01-01T00:00:00Z", "max_input_tokens": 1048576, "max_tokens": 64000, "capabilities": map[string]interface{}{"tools": true, "vision": true}},
	{"id": "sonnet", "type": "model", "display_name": "Claude Sonnet (alias)", "created_at": "2026-01-01T00:00:00Z", "max_input_tokens": 1048576, "max_tokens": 64000, "capabilities": map[string]interface{}{"tools": true, "thinking": true, "vision": true}},
	{"id": "opus", "type": "model", "display_name": "Claude Opus (alias)", "created_at": "2026-01-01T00:00:00Z", "max_input_tokens": 1048576, "max_tokens": 64000, "capabilities": map[string]interface{}{"tools": true, "thinking": true, "vision": true}},
	{"id": "fable", "type": "model", "display_name": "Claude Fable (alias)", "created_at": "2026-01-01T00:00:00Z", "max_input_tokens": 1048576, "max_tokens": 64000, "capabilities": map[string]interface{}{"tools": true, "thinking": true, "vision": true}},
	{"id": "mythos", "type": "model", "display_name": "Claude Mythos (alias)", "created_at": "2026-01-01T00:00:00Z", "max_input_tokens": 1048576, "max_tokens": 64000, "capabilities": map[string]interface{}{"tools": true, "thinking": true, "vision": true}},
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
		// An audit caught that these dynamically-discovered entries
		// (gemini-3.8-flash-high/-medium/-low etc. — the real tiering
		// targets, fetched live from Google) were missing
		// max_input_tokens/max_tokens/capabilities entirely, unlike every
		// static knownModels entry above — an SDK reading /v1/models to
		// decide context budgeting or thinking support for one of these
		// IDs got nothing to go on.
		entry := map[string]interface{}{
			"id":           m.ID,
			"type":         "model",
			"display_name": m.DisplayName,
			"created_at":   "2026-01-01T00:00:00Z",
			"capabilities": map[string]interface{}{"tools": true, "thinking": m.Thinking, "vision": true},
		}
		if m.MaxTokens > 0 {
			entry["max_input_tokens"] = m.MaxTokens
		}
		if m.MaxOutput > 0 {
			entry["max_tokens"] = m.MaxOutput
		}
		out = append(out, entry)
	}
	return out
}

func authenticatedTenantFromRequest(cfgMgr *ConfigManager, r *http.Request) (TenantConfig, bool) {
	clientKey := r.Header.Get("x-api-key")
	if clientKey == "" {
		authHeader := r.Header.Get("Authorization")
		if strings.HasPrefix(strings.ToLower(authHeader), "bearer ") {
			clientKey = strings.TrimSpace(authHeader[7:])
		} else {
			clientKey = authHeader
		}
	}
	return cfgMgr.GetTenant(clientKey)
}

func handleModels(cfgMgr *ConfigManager, w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", "GET")
		writeAnthropicError(w, http.StatusMethodNotAllowed, "invalid_request_error", "This endpoint requires a GET request")
		return
	}
	if _, ok := authenticatedTenantFromRequest(cfgMgr, r); !ok {
		writeAnthropicError(w, http.StatusUnauthorized, "authentication_error", "Invalid or missing API key")
		return
	}
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
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", "GET")
		writeAnthropicError(w, http.StatusMethodNotAllowed, "invalid_request_error", "This endpoint requires a GET request")
		return
	}
	if _, ok := authenticatedTenantFromRequest(cfgMgr, r); !ok {
		writeAnthropicError(w, http.StatusUnauthorized, "authentication_error", "Invalid or missing API key")
		return
	}
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

// isFiveHourWindow and isWeeklyWindow recognize CloudCode's quota-window
// name regardless of separator style. A Round 15 audit caught the plain
// strings.Contains(win, "5h") check (repeated three times below) missed
// "5_hour"/"5-hour" — there's no contiguous "5h" substring in either, so an
// account whose quota JSON happens to use underscore/hyphen separators lost
// its 5-hour bucket match entirely and fell back to the synthetic-quota
// display. Same gap for "week" never matching "7d"/"7_day"/"7-day".
func isFiveHourWindow(win string) bool {
	return strings.Contains(win, "5h") || strings.Contains(win, "5_h") || strings.Contains(win, "5-h") || strings.Contains(win, "18000")
}

func isWeeklyWindow(win string) bool {
	return strings.Contains(win, "week") || strings.Contains(win, "7d") || strings.Contains(win, "7_d") || strings.Contains(win, "7-d")
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
				if isFiveHourWindow(win) {
					fiveHour = b
				} else if isWeeklyWindow(win) {
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
				if isFiveHourWindow(win) {
					fiveHour = b
				} else if isWeeklyWindow(win) {
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
			if isFiveHourWindow(win) {
				fiveHour = b
			} else if isWeeklyWindow(win) {
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
		// Same standard per-window headers the acc != nil path below sends —
		// a generic Anthropic SDK client (not Claude Code, which only reads
		// the unified-* headers above) reads these instead and otherwise
		// gets no rate-limit visibility at all on an api_key tenant.
		const reqLimit, tokLimit = 1000, 1000000
		header.Set("anthropic-ratelimit-requests-limit", fmt.Sprintf("%d", reqLimit))
		header.Set("anthropic-ratelimit-requests-remaining", fmt.Sprintf("%d", int(float64(reqLimit)*0.95)))
		header.Set("anthropic-ratelimit-requests-reset", now.Add(5*time.Hour).UTC().Format(time.RFC3339))
		header.Set("anthropic-ratelimit-tokens-limit", fmt.Sprintf("%d", tokLimit))
		header.Set("anthropic-ratelimit-tokens-remaining", fmt.Sprintf("%d", int(float64(tokLimit)*0.95)))
		header.Set("anthropic-ratelimit-tokens-reset", now.Add(5*time.Hour).UTC().Format(time.RFC3339))
		// An audit caught that only the aggregated tokens-* triplet was
		// ever sent — official Anthropic SDKs and newer Claude Code
		// versions budget input/output tokens separately and read these
		// split headers; without them they have no visibility into either
		// budget specifically.
		setSplitTokenRateLimitHeaders(header, tokLimit, 0.95, now.Add(5*time.Hour))
		header.Set("anthropic-organization-id", fakeOrgID)
		header.Set("anthropic-workspace-id", fakeWorkspaceID)
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

	// Set unconditionally, before the bucket-nil early return below — an
	// audit caught that a response with no quota bucket data at all
	// (fiveHourBucket == nil && weeklyBucket == nil) returned before ever
	// reaching the org/workspace headers set at the end of this function,
	// so that response carried neither — exactly the inconsistency the
	// comment at the bottom of this function says this gateway can't
	// afford to have.
	header.Set("anthropic-organization-id", fakeOrgID)
	header.Set("anthropic-workspace-id", fakeWorkspaceID)

	fiveHourBucket, weeklyBucket := pickFiveHourAndWeekly(acc, modelName)
	if fiveHourBucket == nil && weeklyBucket == nil {
		// A follow-up audit caught that this early return left the
		// response with ZERO anthropic-ratelimit-* headers at all (only
		// org/workspace-id from the fix just above) — the exact same
		// "headers absent" problem the acc == nil branch above already
		// has a synthetic-default answer for (a real account whose
		// quota just hasn't loaded yet, e.g. right after a restart
		// before the async warmup completes, looks no different to the
		// client than one with no account at all; same fix applies).
		now := time.Now()
		header.Set("anthropic-ratelimit-unified-5h-status", "allowed")
		header.Set("anthropic-ratelimit-unified-5h-utilization", "0.05")
		header.Set("anthropic-ratelimit-unified-5h-reset", fmt.Sprintf("%d", now.Add(5*time.Hour).Unix()))
		header.Set("anthropic-ratelimit-unified-status", "allowed")
		header.Set("anthropic-ratelimit-unified-representative-claim", "five_hour")
		header.Set("anthropic-ratelimit-unified-reset", fmt.Sprintf("%d", now.Add(5*time.Hour).Unix()))
		const reqLimit, tokLimit = 1000, 1000000
		header.Set("anthropic-ratelimit-requests-limit", fmt.Sprintf("%d", reqLimit))
		header.Set("anthropic-ratelimit-requests-remaining", fmt.Sprintf("%d", int(float64(reqLimit)*0.95)))
		header.Set("anthropic-ratelimit-requests-reset", now.Add(5*time.Hour).UTC().Format(time.RFC3339))
		header.Set("anthropic-ratelimit-tokens-limit", fmt.Sprintf("%d", tokLimit))
		header.Set("anthropic-ratelimit-tokens-remaining", fmt.Sprintf("%d", int(float64(tokLimit)*0.95)))
		header.Set("anthropic-ratelimit-tokens-reset", now.Add(5*time.Hour).UTC().Format(time.RFC3339))
		setSplitTokenRateLimitHeaders(header, tokLimit, 0.95, now.Add(5*time.Hour))
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
	// A Round 15 audit caught this used >= while the /usage JSON body's
	// own is_active flag (further down in this file, same bucket-fraction
	// comparison) uses fiveHourFrac >= weeklyFrac to decide the OPPOSITE
	// tie-break — on an exact tie, the header said "seven_day" while the
	// JSON body marked five_hour as is_active:true, desyncing the UI.
	// Strict > keeps this consistent with that is_active logic.
	claim, topStatus, topReset := "five_hour", status(fiveHourFrac), fiveHourReset
	if weeklyBucket != nil && weeklyFrac > fiveHourFrac {
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
	// An audit caught that gating this whole block on fiveHourBucket alone
	// meant an account with ONLY a weekly bucket (no 5h data parsed, or
	// genuinely 5h-exhausted-and-expired while the week isn't) got NONE
	// of these standard headers at all — not degraded, just silently
	// absent, violating the exact "every successful response carries
	// these" contract this comment itself is about. Fall back to the
	// weekly bucket's frac/reset when the 5h one isn't available.
	standardFrac, standardReset, haveStandard := fiveHourFrac, fiveHourReset, fiveHourBucket != nil
	if !haveStandard && weeklyBucket != nil {
		standardFrac, standardReset, haveStandard = weeklyFrac, weeklyReset, true
	}
	if haveStandard {
		const reqLimit, tokLimit = 1000, 1000000
		header.Set("anthropic-ratelimit-requests-limit", fmt.Sprintf("%d", reqLimit))
		header.Set("anthropic-ratelimit-requests-remaining", fmt.Sprintf("%d", int(float64(reqLimit)*(1-standardFrac))))
		header.Set("anthropic-ratelimit-requests-reset", time.Unix(standardReset, 0).UTC().Format(time.RFC3339))
		header.Set("anthropic-ratelimit-tokens-limit", fmt.Sprintf("%d", tokLimit))
		header.Set("anthropic-ratelimit-tokens-remaining", fmt.Sprintf("%d", int(float64(tokLimit)*(1-standardFrac))))
		header.Set("anthropic-ratelimit-tokens-reset", time.Unix(standardReset, 0).UTC().Format(time.RFC3339))
		setSplitTokenRateLimitHeaders(header, tokLimit, 1-standardFrac, time.Unix(standardReset, 0))
	}

	// anthropic-organization-id/anthropic-workspace-id are set earlier in
	// this function, before the bucket-nil early return — see the comment
	// there for why.
}

const (
	fakeOrgID       = "0bad747f-0000-4000-a000-000000000000"
	fakeWorkspaceID = "wrkspc_0000000000000000000000000"
	// An audit caught that several hardcoded CLI-spoof responses used
	// this SAME org UUID for the account/user object too — real Claude
	// clients can cache fetched objects keyed by UUID, so a client
	// fetching "account" and "organization" back to back saw identical
	// UUIDs for two conceptually different entities, risking one
	// overwriting the other's cached fields.
	fakeAccountID = "0bad747f-0000-4000-a000-000000000001"
)

func handleUsage(cfgMgr *ConfigManager, w http.ResponseWriter, r *http.Request) {
	clientKey := r.Header.Get("x-api-key")
	if clientKey == "" {
		authHeader := r.Header.Get("Authorization")
		if strings.HasPrefix(strings.ToLower(authHeader), "bearer ") {
			clientKey = strings.TrimSpace(authHeader[7:])
		} else {
			clientKey = authHeader
		}
	}
	tenant, _ := cfgMgr.GetTenant(clientKey)

	convAnchor := r.URL.Query().Get("conversation")
	if convAnchor == "" {
		// An audit caught this only ever looked at a "conversation" query
		// parameter — Claude Code's own CLI identifies a session via the
		// x-claude-code-session-id HEADER on /v1/messages (see
		// DeriveConversationAnchor's real caller), not a query string on
		// this GET /usage-style endpoint, so a usage query for a specific
		// session fell straight through to the generic per-API-key sticky
		// fallback below instead of resolving to that session's actual
		// bound account, returning the WRONG account's quota data.
		if headerSessionID := r.Header.Get("x-claude-code-session-id"); headerSessionID != "" {
			convAnchor = DeriveConversationAnchor(nil, headerSessionID, "")
		}
	}
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

	// An audit caught this 404'd outright for a tenant with
	// google_auth_type "api_key" (no OAuth pool at all, so acc is nil) —
	// handleOAuthUsage already has a synthetic low-utilization fallback
	// for this exact case (isSyntheticAccount), but this sibling endpoint
	// never got the same treatment, so /v1/usage was simply broken for
	// every API-key-only tenant instead of showing a stable low number.
	if acc == nil {
		now := time.Now()
		resp := map[string]interface{}{
			"account":        "",
			"project_id":     "",
			"cooldown_until": nil,
			"last_checked":   now,
			"groups": []GroupOut{{
				DisplayName: "Usage",
				Description: "This tenant authenticates with a direct API key, not a Google OAuth account pool — no real quota data to report.",
				Buckets: []BucketOut{
					{Window: "5h", DisplayName: "5-Hour", RemainingPercent: 95.0, ResetTime: now.Add(5 * time.Hour).UTC().Format(time.RFC3339), Description: "Synthetic — no OAuth account pool"},
					{Window: "7d", DisplayName: "Weekly", RemainingPercent: 95.0, ResetTime: now.Add(7 * 24 * time.Hour).UTC().Format(time.RFC3339), Description: "Synthetic — no OAuth account pool"},
				},
			}},
		}
		b, _ := json.MarshalIndent(resp, "", "  ")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(b)
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

// handleDeprecatedComplete answers the legacy Anthropic text-completions
// endpoint (pre-Messages-API) with a clear, actionable error instead of
// letting it fall through to the dashboard's Basic-Auth catch-all.
func handleDeprecatedComplete(w http.ResponseWriter, r *http.Request) {
	writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "The /v1/complete endpoint is deprecated and not supported by this gateway; use /v1/messages instead.")
}

// handleUnsupportedBatches answers the Message Batches API with a clear
// "not supported" instead of a 401 Basic Auth challenge — this gateway has
// no batch-processing backend to run it against.
func handleUnsupportedBatches(w http.ResponseWriter, r *http.Request) {
	writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "Message batches are not supported by this gateway.")
}

// handleUnknownV1Path is the last-resort fallback for any /v1/* path not
// matched by a more specific handler above (ServeMux always prefers the
// most specific registered pattern) — a typo'd or unimplemented endpoint
// now gets a clean Anthropic 404 instead of the dashboard's HTML/Basic-Auth
// catch-all.
func handleUnknownV1Path(w http.ResponseWriter, r *http.Request) {
	writeAnthropicError(w, http.StatusNotFound, "not_found_error", fmt.Sprintf("Unknown endpoint: %s %s", r.Method, r.URL.Path))
}

func handleEmbeddings(cfgMgr *ConfigManager, w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeAnthropicError(w, http.StatusMethodNotAllowed, "invalid_request_error", "Method Not Allowed")
		return
	}

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "Bad Request")
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
		if strings.HasPrefix(strings.ToLower(authHeader), "bearer ") {
			clientKey = strings.TrimSpace(authHeader[7:])
		} else {
			clientKey = authHeader
		}
	}
	tenant, _ := cfgMgr.GetTenant(clientKey)

	convAnchor := r.URL.Query().Get("conversation")
	if convAnchor == "" {
		// Same session-header fallback as handleUsage above — see its
		// comment for the full explanation.
		if headerSessionID := r.Header.Get("x-claude-code-session-id"); headerSessionID != "" {
			convAnchor = DeriveConversationAnchor(nil, headerSessionID, "")
		}
	}
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

	// A tenant on google_auth_type "api_key" has no OAuth pool at all, so
	// acc is nil here — this used to 404, breaking the /usage command
	// outright instead of showing a low, stable utilization the same way
	// applyRateLimitHeaders' own synthetic values already do for this
	// exact case. isSyntheticAccount skips the real quota lookups below
	// and falls through to the same resp-building code with synthetic
	// windows, instead of duplicating that whole structure by hand.
	isSyntheticAccount := acc == nil

	// Only block on a real network round-trip the FIRST time this account
	// has no quota data at all (matching applyRateLimitHeaders' own
	// first-fetch rule) — every call after that used to re-fetch
	// synchronously on every single /usage poll, so the command hung for
	// however long Google's quota endpoint took (seconds, under any load).
	// Serve the cached snapshot immediately and let the same
	// 45s-staleness async refresh everyone else already uses catch it up
	// in the background.
	var fiveHourBucket, weeklyBucket *QuotaBucket
	var fiveHourWindow, weeklyWindow map[string]interface{}
	if isSyntheticAccount {
		now := time.Now()
		synthWindow := func(reset time.Time) map[string]interface{} {
			return map[string]interface{}{
				"utilization":       5.0,
				"resets_at":         reset.UTC().Format("2006-01-02T15:04:05.000Z"),
				"limit_dollars":     nil,
				"used_dollars":      nil,
				"remaining_dollars": nil,
				"locked_reason":     nil,
			}
		}
		fiveHourWindow = synthWindow(now.Add(5 * time.Hour))
		weeklyWindow = synthWindow(now.Add(7 * 24 * time.Hour))
	} else {
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
		fiveHourBucket, weeklyBucket = pickFiveHourAndWeekly(acc, "gemini-3.8-flash-tiered")
		fiveHourWindow = bucketToUsageWindow(fiveHourBucket)
		weeklyWindow = bucketToUsageWindow(weeklyBucket)
	}

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
		// Provide model-scoped weekly limits so Claude Code can render
		// individual progress bars for Opus and Sonnet tiers in /usage
		limits = append(limits, map[string]interface{}{
			"kind":      "weekly_scoped",
			"group":     "weekly",
			"percent":   weeklyWindow["utilization"],
			"severity":  severityFor(weeklyFrac),
			"resets_at": weeklyWindow["resets_at"],
			"scope": map[string]interface{}{
				"model": map[string]interface{}{
					"display_name": "Opus",
				},
			},
			"is_active": false,
		})
		limits = append(limits, map[string]interface{}{
			"kind":      "weekly_scoped",
			"group":     "weekly",
			"percent":   weeklyWindow["utilization"],
			"severity":  severityFor(weeklyFrac),
			"resets_at": weeklyWindow["resets_at"],
			"scope": map[string]interface{}{
				"model": map[string]interface{}{
					"display_name": "Sonnet",
				},
			},
			"is_active": false,
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
			"monthly_limit":        0,
			"used_credits":         0,
			"utilization":          0,
			"currency":             "USD",
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
	applyRateLimitHeaders(w.Header(), acc, "gemini-3.8-flash-tiered")
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write(b)
}
