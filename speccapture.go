// This file captures the STRUCTURE (not the generative content) of every
// real request/response that passes through the gateway, so a separate
// offline tool (cmd/speccheck) can statically check real traffic against a
// growing list of specification invariants instead of someone reasoning
// about Go code and manually reproducing a bug with curl every time a
// regression shows up — see AGENT_HANDOFF_NOTES.md's "live production
// breaks" list for the incidents that motivated this.
package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// specRedactThreshold is the only rule deciding what counts as "generative
// text" worth stripping: any JSON string longer than this many characters
// is almost certainly prose, a tool-result body, or a base64 blob — never a
// role/type/model/tool name, a stop_reason, a header value, or an id, all
// of which are comfortably shorter. A hardcoded per-field name/path
// allowlist would need to be kept in sync with both Claude's and Gemini's
// evolving schemas forever; a length threshold doesn't.
const specRedactThreshold = 200

// redactSkeleton recursively strips long string VALUES, keeping every map
// key, every short string (the actual structure/enums/ids we care about),
// every number, bool, and null, and the shape (object/array nesting, array
// lengths) fully intact.
func redactSkeleton(v interface{}) interface{} {
	switch t := v.(type) {
	case map[string]interface{}:
		out := make(map[string]interface{}, len(t))
		for k, val := range t {
			out[k] = redactSkeleton(val)
		}
		return out
	case []interface{}:
		out := make([]interface{}, len(t))
		for i, val := range t {
			out[i] = redactSkeleton(val)
		}
		return out
	case string:
		if len(t) > specRedactThreshold {
			return fmt.Sprintf("<redacted text:%d chars>", len(t))
		}
		return t
	default:
		return v
	}
}

// redactJSONBytes parses raw JSON and redacts it. A body that fails to
// parse (never expected, but this must never panic/crash a live request
// over a capture-side bug) is recorded as just its length.
func redactJSONBytes(raw []byte) interface{} {
	if len(raw) == 0 {
		return nil
	}
	var v interface{}
	if err := json.Unmarshal(raw, &v); err != nil {
		return map[string]interface{}{"_unparsed_len": len(raw)}
	}
	return redactSkeleton(v)
}

// isSpecSensitiveHeader flags headers whose VALUE is a credential, never
// structure — these get stored as presence+length only, same spirit as
// request_logger.go's MaskKey for the dashboard's own tenant-key display.
func isSpecSensitiveHeader(name string) bool {
	l := strings.ToLower(name)
	if l == "authorization" || l == "x-api-key" || l == "proxy-authorization" {
		return true
	}
	// Deliberately NOT a bare "token" substring match — that also matches
	// the entirely non-sensitive anthropic-ratelimit-tokens-limit/
	// -remaining/-reset headers (rate-limit NUMBERS, not credentials),
	// hiding exactly the structural data this capture exists to keep.
	return strings.Contains(l, "secret") || strings.Contains(l, "access-token") ||
		strings.Contains(l, "refresh-token") || strings.Contains(l, "bearer")
}

func redactHeaders(h http.Header) map[string]string {
	if len(h) == 0 {
		return nil
	}
	out := make(map[string]string, len(h))
	for k, v := range h {
		val := strings.Join(v, ", ")
		if isSpecSensitiveHeader(k) {
			out[k] = fmt.Sprintf("<present, %d chars>", len(val))
			continue
		}
		out[k] = val
	}
	return out
}

// SpecSnapshot is one captured leg (a request or a response) of the
// four-leg round trip this gateway sits in the middle of.
type SpecSnapshot struct {
	Headers map[string]string `json:"headers,omitempty"`
	Body    interface{}       `json:"body,omitempty"`
}

// inFlightCapture accumulates the four legs of one request while it's
// still being processed. The streaming legs (geminiChunks/claudeEvents)
// are appended to incrementally from sse.go, since a streamed response
// never exists as one complete body the way a non-streaming one does.
type inFlightCapture struct {
	rid       uint64
	timestamp time.Time
	sessionID string
	agentKey  string

	claudeRequest  *SpecSnapshot
	geminiRequest  *SpecSnapshot
	geminiResponse *SpecSnapshot
	claudeResponse *SpecSnapshot

	geminiChunks []interface{}
	claudeEvents []interface{}
}

// SpecCaptureStore is the package-level, mutex-guarded store every capture
// call goes through — same shape as globalLogStore/globalModelCatalog/
// deadRefreshTokens elsewhere in this codebase, chosen specifically so no
// function signature across server.go/sse.go needs a new parameter
// threaded through just to reach this; every call site already has the
// request's `rid` in scope.
type SpecCaptureStore struct {
	mu       sync.Mutex
	enabled  bool
	dir      string
	retain   int
	inFlight map[uint64]*inFlightCapture

	fileMu   sync.Mutex
	fileDate string
	file     *os.File
}

var globalSpecCapture = &SpecCaptureStore{inFlight: make(map[uint64]*inFlightCapture)}

// InitSpecCapture wires up the store from config at startup. Safe to call
// with enabled=false — every method below becomes a near-zero-cost no-op.
func InitSpecCapture(enabled bool, dir string, retentionDays int) {
	if dir == "" {
		dir = "request_captures"
	}
	globalSpecCapture.mu.Lock()
	globalSpecCapture.enabled = enabled
	globalSpecCapture.dir = dir
	globalSpecCapture.retain = retentionDays
	globalSpecCapture.mu.Unlock()
	if !enabled {
		return
	}
	if err := os.MkdirAll(dir, 0755); err != nil {
		LogGateway("[SuperRelay Gateway] Warning: spec capture disabled, could not create %s: %v", dir, err)
		globalSpecCapture.mu.Lock()
		globalSpecCapture.enabled = false
		globalSpecCapture.mu.Unlock()
		return
	}
	globalSpecCapture.pruneOld()
}

func (s *SpecCaptureStore) Begin(rid uint64) {
	if !s.enabled {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.inFlight[rid] = &inFlightCapture{rid: rid, timestamp: time.Now()}
}

func (s *SpecCaptureStore) get(rid uint64) *inFlightCapture {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.inFlight[rid]
}

// SetSessionMeta attaches the gateway's own session/agent identifiers —
// the same ones the dashboard already shows per request — to the capture
// so `cmd/speccheck -session <id>` can find every request belonging to
// one conversation. Called once sessionID is known (shortly before
// translation, later than Begin), so it's a separate method rather than
// a Begin parameter.
func (s *SpecCaptureStore) SetSessionMeta(rid uint64, sessionID, agentKey string) {
	if !s.enabled {
		return
	}
	if c := s.get(rid); c != nil {
		c.sessionID = sessionID
		c.agentKey = agentKey
	}
}

func (s *SpecCaptureStore) RecordClaudeRequest(rid uint64, headers http.Header, body []byte) {
	if !s.enabled {
		return
	}
	if c := s.get(rid); c != nil {
		c.claudeRequest = &SpecSnapshot{Headers: redactHeaders(headers), Body: redactJSONBytes(body)}
	}
}

func (s *SpecCaptureStore) RecordGeminiRequest(rid uint64, headers http.Header, body []byte) {
	if !s.enabled {
		return
	}
	if c := s.get(rid); c != nil {
		c.geminiRequest = &SpecSnapshot{Headers: redactHeaders(headers), Body: redactJSONBytes(body)}
	}
}

func (s *SpecCaptureStore) RecordGeminiResponse(rid uint64, headers http.Header, body []byte) {
	if !s.enabled {
		return
	}
	if c := s.get(rid); c != nil {
		c.geminiResponse = &SpecSnapshot{Headers: redactHeaders(headers), Body: redactJSONBytes(body)}
	}
}

func (s *SpecCaptureStore) RecordClaudeResponse(rid uint64, headers http.Header, body interface{}) {
	if !s.enabled {
		return
	}
	if c := s.get(rid); c != nil {
		var redacted interface{}
		if raw, ok := body.([]byte); ok {
			redacted = redactJSONBytes(raw)
		} else {
			redacted = redactSkeleton(body)
		}
		c.claudeResponse = &SpecSnapshot{Headers: redactHeaders(headers), Body: redacted}
	}
}

// AppendGeminiChunk records one streamed Gemini SSE/JSON chunk. Called from
// sse.go's ProcessGeminiChunk — one hook covers every chunk shape Gemini
// can send.
func (s *SpecCaptureStore) AppendGeminiChunk(rid uint64, chunk []byte) {
	if !s.enabled {
		return
	}
	if c := s.get(rid); c != nil {
		c.geminiChunks = append(c.geminiChunks, redactJSONBytes(chunk))
	}
}

// AppendClaudeEvent records one outbound Anthropic SSE event. Called from
// SSEStreamer.SendRaw — the single lowest-level emit point every event
// type (message_start, content_block_delta, message_stop, ...) already
// passes through, so this is one hook for the whole event vocabulary.
func (s *SpecCaptureStore) AppendClaudeEvent(rid uint64, event string, data string) {
	if !s.enabled {
		return
	}
	c := s.get(rid)
	if c == nil {
		return
	}
	rec := map[string]interface{}{"event": event}
	if parsed := redactJSONBytes([]byte(data)); parsed != nil {
		rec["data"] = parsed
	}
	c.claudeEvents = append(c.claudeEvents, rec)
}

// Finish assembles whatever legs were recorded into one NDJSON line and
// drops the in-memory entry. Safe to call unconditionally on every exit
// path of a request (including ones that never called Begin, or that
// began but errored out before any other leg was recorded) — it's a no-op
// whenever there's nothing to flush.
func (s *SpecCaptureStore) Finish(rid uint64) {
	if !s.enabled {
		return
	}
	s.mu.Lock()
	c, ok := s.inFlight[rid]
	if ok {
		delete(s.inFlight, rid)
	}
	s.mu.Unlock()
	if !ok {
		return
	}

	out := map[string]interface{}{
		"rid":       c.rid,
		"timestamp": c.timestamp.UTC().Format(time.RFC3339Nano),
	}
	if c.sessionID != "" {
		out["session_id"] = c.sessionID
	}
	if c.agentKey != "" {
		out["agent_key"] = c.agentKey
	}
	if c.claudeRequest != nil {
		out["claude_request"] = c.claudeRequest
	}
	if c.geminiRequest != nil {
		out["gemini_request"] = c.geminiRequest
	}
	if c.geminiResponse != nil {
		out["gemini_response"] = c.geminiResponse
	} else if len(c.geminiChunks) > 0 {
		out["gemini_response"] = map[string]interface{}{"chunks": c.geminiChunks}
	}
	if c.claudeResponse != nil {
		out["claude_response"] = c.claudeResponse
	} else if len(c.claudeEvents) > 0 {
		out["claude_response"] = map[string]interface{}{"events": c.claudeEvents}
	}

	// Nothing but rid/timestamp/session/agent-key got recorded (e.g. an
	// auth failure before Begin's caller ever reached a capture point) —
	// not worth a line.
	_, hasClaudeReq := out["claude_request"]
	_, hasGeminiReq := out["gemini_request"]
	_, hasGeminiResp := out["gemini_response"]
	_, hasClaudeResp := out["claude_response"]
	if !hasClaudeReq && !hasGeminiReq && !hasGeminiResp && !hasClaudeResp {
		return
	}

	line, err := json.Marshal(out)
	if err != nil {
		return
	}
	s.writeLine(line)
}

func (s *SpecCaptureStore) writeLine(line []byte) {
	s.fileMu.Lock()
	defer s.fileMu.Unlock()
	today := time.Now().UTC().Format("2006-01-02")
	if s.file == nil || s.fileDate != today {
		if s.file != nil {
			_ = s.file.Close()
		}
		path := filepath.Join(s.dir, today+".ndjson")
		f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			LogGateway("[SuperRelay Gateway] Warning: spec capture could not open %s: %v", path, err)
			return
		}
		s.file = f
		s.fileDate = today
		s.pruneOld()
	}
	line = append(line, '\n')
	_, _ = s.file.Write(line)
}

// pruneOld deletes capture files older than the configured retention,
// going by the date encoded in the filename rather than mtime — simpler,
// and correct even if the files were copied/touched by something else.
func (s *SpecCaptureStore) pruneOld() {
	if s.retain <= 0 {
		return
	}
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return
	}
	cutoff := time.Now().UTC().AddDate(0, 0, -s.retain)
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		dateStr, isCapture := strings.CutSuffix(name, ".ndjson")
		if !isCapture {
			continue
		}
		t, err := time.Parse("2006-01-02", dateStr)
		if err != nil {
			continue
		}
		if t.Before(cutoff) {
			_ = os.Remove(filepath.Join(s.dir, name))
		}
	}
}
