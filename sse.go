package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// toolCallSeq makes generated tool_use ids collision-free regardless of
// timer resolution: time.Now().UnixNano() alone can return the same value
// for two tool calls processed microseconds apart (common on Windows/VMs
// with coarse clock granularity), which broke the Anthropic API's
// requirement that every tool_use id in a message be unique.
var toolCallSeq atomic.Uint64

// nextToolCallID builds a tool_use id of the form
// "toolu_<name>_<nanots>_<seq>" (see translator.go's recovery parse of this
// exact format). Anthropic's real tool_use.id pattern is
// ^[a-zA-Z0-9_-]{1,64}$ — a long MCP tool name ("mcp__server__some-
// descriptive-tool-name") plus a decimal nanosecond timestamp routinely
// exceeded 64 characters, which a strict client SDK rejects outright.
// Hex-encoding the numeric parts (shorter than decimal, still digits for
// the trailing-segment recovery check — see isAllHexDigits) buys back some
// room; the name itself is truncated as a last resort only if it still
// doesn't fit, which only degrades that same best-effort recovery path for
// an already-rare case, never the live tool call itself (name/args are
// carried separately in the SSE/JSON payload, not reconstructed from the
// id).
func nextToolCallID(name string) string {
	safeName := strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' || r == '-' {
			return r
		}
		return '_'
	}, name)
	nanoHex := fmt.Sprintf("%x", time.Now().UnixNano())
	seqHex := fmt.Sprintf("%x", toolCallSeq.Add(1))

	const prefix = "toolu_"
	fixedLen := len(prefix) + len(nanoHex) + len(seqHex) + 2 // 2 separators
	maxNameLen := 64 - fixedLen
	if maxNameLen < 1 {
		maxNameLen = 1
	}
	if len(safeName) > maxNameLen {
		safeName = safeName[:maxNameLen]
	}
	id := fmt.Sprintf("%s%s_%s_%s", prefix, safeName, nanoHex, seqHex)
	if len(id) > 64 {
		id = id[:64]
	}
	return id
}

const (
	toolSignatureStaleAfter = 2 * time.Hour
	toolSignatureSweepEvery = 30 * time.Minute
)

// toolSignatureCache remembers the real Gemini thoughtSignature attached to
// a functionCall part, keyed by that call's own tool_use id. Anthropic's
// tool_use content block has no signature field to round-trip it through
// Claude Code itself (unlike "thinking" blocks, which do) — this is the only
// way to recover the real signature when that same tool_use id is echoed
// back on a later turn, instead of always falling back to Google's
// "skip_thought_signature_validator" bypass, which its own docs warn
// degrades reasoning quality across multi-turn tool-calling chains.
var (
	toolSignatureMu   sync.Mutex
	toolSignatureMap  = make(map[string]string)
	toolSignatureSeen = make(map[string]time.Time)
)

func init() {
	go func() {
		ticker := time.NewTicker(toolSignatureSweepEvery)
		defer ticker.Stop()
		for range ticker.C {
			sweepStaleToolSignatures()
		}
	}()
}

func sweepStaleToolSignatures() {
	toolSignatureMu.Lock()
	defer toolSignatureMu.Unlock()
	cutoff := time.Now().Add(-toolSignatureStaleAfter)
	for id, t := range toolSignatureSeen {
		if t.Before(cutoff) {
			delete(toolSignatureSeen, id)
			delete(toolSignatureMap, id)
		}
	}
}

func setToolSignature(callID, signature string) {
	if callID == "" || signature == "" {
		return
	}
	toolSignatureMu.Lock()
	defer toolSignatureMu.Unlock()
	toolSignatureMap[callID] = signature
	toolSignatureSeen[callID] = time.Now()
}

func getToolSignature(callID string) string {
	toolSignatureMu.Lock()
	defer toolSignatureMu.Unlock()
	return toolSignatureMap[callID]
}

type SSEStreamer struct {
	w                    http.ResponseWriter
	flusher              http.Flusher
	mu                   sync.Mutex
	hasStarted           bool
	currentType          int // 0: none, 1: text, 2: thinking, 3: tool_use
	currentIndex         int
	messageID            string
	model                string
	inputTokens          int64
	outputTokens         int64
	cacheTokens          int64
	toolUseOccurred      bool
	textChars            int
	thinkingChars        int
	toolNames            []string
	toolCallSignatures   []string
	lastText             string
	lastEventTime        time.Time
	searchQueries        []string
	groundingSources     []GroundingSource
	sessionID            string
	requestedSafeguards  bool
	toolCallIDs          []string
	hasFinished          bool
	thoughtSignature     string          // most recent thoughtSignature seen for the open thinking block, flushed as signature_delta on close
	estInputTokens       int             // approximate input token count, known upfront unlike Gemini's real usageMetadata (only arrives at stream end)
	preferRussian        bool            // conversation's own language, detected from the user's latest message — picks which language gateway-synthesized text (grounding footer) uses, since that text bypasses the model entirely
	syntheticReasonTools map[string]bool // tool names whose Gemini-bound schema got our synthetic "reason" placeholder injected (translator.go); that key is stripped back out of the model's actual function-call args before handing them to Claude Code
}

type GroundingSource struct {
	Title string `json:"title"`
	URI   string `json:"uri"`
}

const (
	TypeNone     = 0
	TypeText     = 1
	TypeThinking = 2
	TypeToolUse  = 3
)

func NewSSEStreamer(w http.ResponseWriter, flusher http.Flusher, model string, sessionID string, requestedSafeguards bool, estInputTokens int, preferRussian bool, syntheticReasonTools map[string]bool) *SSEStreamer {
	return &SSEStreamer{
		w:                    w,
		flusher:              flusher,
		messageID:            fmt.Sprintf("msg_%d", time.Now().UnixNano()),
		model:                model,
		currentIndex:         -1,
		lastEventTime:        time.Now(),
		sessionID:            sessionID,
		requestedSafeguards:  requestedSafeguards,
		estInputTokens:       estInputTokens,
		preferRussian:        preferRussian,
		syntheticReasonTools: syntheticReasonTools,
	}
}

func (s *SSEStreamer) SendRaw(event, data string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastEventTime = time.Now()
	msg := fmt.Sprintf("event: %s\ndata: %s\n\n", event, data)
	_, err := s.w.Write([]byte(msg))
	if err == nil && s.flusher != nil {
		s.flusher.Flush()
	}
	return err
}

func (s *SSEStreamer) SendKeepalivePing() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if time.Since(s.lastEventTime) < 4*time.Second {
		return nil
	}
	s.lastEventTime = time.Now()
	_, err := s.w.Write([]byte("event: ping\ndata: {\"type\":\"ping\"}\n\n"))
	if err == nil && s.flusher != nil {
		s.flusher.Flush()
	}
	return err
}

func (s *SSEStreamer) EnsureMessageStart() error {
	if s.hasStarted {
		return nil
	}
	s.hasStarted = true
	startPayload := map[string]interface{}{
		"type": "message_start",
		"message": map[string]interface{}{
			"id":            s.messageID,
			"type":          "message",
			"role":          "assistant",
			"content":       []interface{}{},
			"model":         s.model,
			"stop_reason":   nil,
			"stop_sequence": nil,
			// Gemini's own usageMetadata (real cache split) only arrives in
			// the stream's last chunk, so these two can't be predicted
			// honestly this early — present as the documented shape with
			// 0 rather than omitted, in case a strict SDK expects the
			// fields to exist at message_start even before they're
			// corrected via message_delta.
			"usage": map[string]interface{}{
				"input_tokens":                s.estInputTokens,
				"output_tokens":               0,
				"cache_creation_input_tokens": 0,
				"cache_read_input_tokens":     0,
			},
		},
	}
	b, _ := json.Marshal(startPayload)
	return s.SendRaw("message_start", string(b))
}

func (s *SSEStreamer) closeCurrentBlock() error {
	if s.currentType == TypeNone || s.currentIndex < 0 {
		return nil
	}
	// Anthropic's extended-thinking streaming format sends a signature_delta
	// closing out the thinking block's crypto signature before
	// content_block_stop; skipping it silently is what pushed us toward the
	// "skip_thought_signature_validator" placeholder on the translate-back
	// path (translator.go) in the first place, since the real signature was
	// never captured and forwarded. Flush whatever we saw.
	if s.currentType == TypeThinking {
		sig := s.thoughtSignature
		if sig == "" {
			// A thinking block with no signature at all isn't a smaller
			// problem than a wrong one — Anthropic's schema declares
			// "signature" required on every thinking block, and an SDK's
			// Pydantic/Zod model raises a validation error on a missing
			// field, not just a mismatched one. Gemini not returning a
			// thoughtSignature for this turn (happens on the public API and
			// some model versions) used to skip signature_delta entirely;
			// fall back to the same placeholder translator.go already uses
			// when replaying history with no real signature to round-trip.
			sig = "skip_thought_signature_validator"
		}
		sigPayload := map[string]interface{}{
			"type":  "content_block_delta",
			"index": s.currentIndex,
			"delta": map[string]interface{}{
				"type":      "signature_delta",
				"signature": sig,
			},
		}
		b, _ := json.Marshal(sigPayload)
		if err := s.SendRaw("content_block_delta", string(b)); err != nil {
			return err
		}
	}
	s.thoughtSignature = ""
	stopPayload := map[string]interface{}{
		"type":  "content_block_stop",
		"index": s.currentIndex,
	}
	b, _ := json.Marshal(stopPayload)
	s.currentType = TypeNone
	return s.SendRaw("content_block_stop", string(b))
}

func (s *SSEStreamer) HandleThinking(text string, thoughtSignature string) error {
	if err := s.EnsureMessageStart(); err != nil {
		return err
	}
	if s.currentType != TypeThinking {
		if err := s.closeCurrentBlock(); err != nil {
			return err
		}
		s.currentIndex++
		s.currentType = TypeThinking

		startPayload := map[string]interface{}{
			"type":  "content_block_start",
			"index": s.currentIndex,
			"content_block": map[string]interface{}{
				"type":     "thinking",
				"thinking": "",
			},
		}
		b, _ := json.Marshal(startPayload)
		if err := s.SendRaw("content_block_start", string(b)); err != nil {
			return err
		}
	}
	if thoughtSignature != "" {
		s.thoughtSignature = thoughtSignature
	}

	deltaPayload := map[string]interface{}{
		"type":  "content_block_delta",
		"index": s.currentIndex,
		"delta": map[string]interface{}{
			"type":     "thinking_delta",
			"thinking": text,
		},
	}
	s.thinkingChars += len(text)
	b, _ := json.Marshal(deltaPayload)
	return s.SendRaw("content_block_delta", string(b))
}

func (s *SSEStreamer) HandleText(text string) error {
	if err := s.EnsureMessageStart(); err != nil {
		return err
	}
	if s.currentType != TypeText {
		if err := s.closeCurrentBlock(); err != nil {
			return err
		}
		s.currentIndex++
		s.currentType = TypeText

		startPayload := map[string]interface{}{
			"type":  "content_block_start",
			"index": s.currentIndex,
			"content_block": map[string]interface{}{
				"type": "text",
				"text": "",
			},
		}
		b, _ := json.Marshal(startPayload)
		if err := s.SendRaw("content_block_start", string(b)); err != nil {
			return err
		}
	}

	deltaPayload := map[string]interface{}{
		"type":  "content_block_delta",
		"index": s.currentIndex,
		"delta": map[string]interface{}{
			"type": "text_delta",
			"text": text,
		},
	}
	s.textChars += len(text)
	s.lastText += text
	b, _ := json.Marshal(deltaPayload)
	return s.SendRaw("content_block_delta", string(b))
}

func (s *SSEStreamer) HandleFunctionCall(name string, argsRaw string, thoughtSignature string) error {
	argsRaw = coerceNumericEnumArgs(name, argsRaw)
	if repaired, ok := RepairEditToolCall(name, argsRaw, s.sessionID); ok {
		argsRaw = repaired
	}
	switch strings.ToLower(name) {
	case "grep":
		if repaired, ok := RepairGrepToolCall(argsRaw); ok {
			argsRaw = repaired
		}
	case "glob":
		if repaired, ok := RepairGlobToolCall(argsRaw); ok {
			argsRaw = repaired
		}
	}
	if s.syntheticReasonTools[name] {
		// This tool's Gemini-bound schema only has a "reason" property
		// because its real schema had none at all (addEmptySchemaPlaceholder
		// in translator.go) — the model dutifully filled it in since it's
		// marked required, but it was never part of the tool's actual
		// declared arguments, and a strict handler (additionalProperties:
		// false) rejects an argument it never declared.
		if cleaned, err := sjson.Delete(argsRaw, "reason"); err == nil {
			argsRaw = cleaned
		}
	}
	argsRaw = stripNestedOpenDictReason(name, argsRaw)
	if err := s.EnsureMessageStart(); err != nil {
		return err
	}
	// Gemini 2.0/3.x Flash Thinking often attaches the open thinking block's
	// thoughtSignature to the chunk carrying the functionCall itself rather
	// than to a preceding thought chunk — if so, s.thoughtSignature is still
	// empty here and closeCurrentBlock()'s signature_delta check below would
	// silently drop it, closing the thinking block with no signature at all.
	if s.currentType == TypeThinking && s.thoughtSignature == "" && thoughtSignature != "" {
		s.thoughtSignature = thoughtSignature
	}
	if err := s.closeCurrentBlock(); err != nil {
		return err
	}
	s.currentIndex++
	s.currentType = TypeToolUse
	s.toolUseOccurred = true
	s.toolNames = append(s.toolNames, name)
	s.toolCallSignatures = append(s.toolCallSignatures, fmt.Sprintf("%s:%s", name, argsRaw))

	callID := nextToolCallID(name)
	s.toolCallIDs = append(s.toolCallIDs, callID)
	setToolSignature(callID, thoughtSignature)

	startPayload := map[string]interface{}{
		"type":  "content_block_start",
		"index": s.currentIndex,
		"content_block": map[string]interface{}{
			"type":  "tool_use",
			"id":    callID,
			"name":  name,
			"input": map[string]interface{}{},
		},
	}
	b, _ := json.Marshal(startPayload)
	if err := s.SendRaw("content_block_start", string(b)); err != nil {
		return err
	}

	deltaPayload := map[string]interface{}{
		"type":  "content_block_delta",
		"index": s.currentIndex,
		"delta": map[string]interface{}{
			"type":         "input_json_delta",
			"partial_json": argsRaw,
		},
	}
	b, _ = json.Marshal(deltaPayload)
	if err := s.SendRaw("content_block_delta", string(b)); err != nil {
		return err
	}

	return s.closeCurrentBlock()
}

// FinishWithError is the fallback for when the upstream read loop
// (StreamGeminiToClaude) dies mid-stream on something other than a clean
// EOF (network drop, a Read error) — previously that error just propagated
// up to be logged, with no event:error/message_stop ever sent on a
// connection that had already committed message_start and possibly several
// content blocks. The client has no way to know the response is over and
// hangs waiting for a message_stop that never arrives (until its own idle
// timeout, if any). Best-effort: if the connection is well and truly dead
// these writes fail too and are ignored, same as they would be anywhere
// else in this file.
func (s *SSEStreamer) FinishWithError(err error) {
	s.mu.Lock()
	finished := s.hasFinished
	s.hasFinished = true
	s.mu.Unlock()
	if finished {
		return
	}
	errPayload := fmt.Sprintf(`{"type":"error","error":{"type":"api_error","message":%q}}`, "Stream connection broken: "+err.Error())
	_ = s.SendRaw("error", errPayload)
	_ = s.SendRaw("message_stop", `{"type":"message_stop"}`)
}

func (s *SSEStreamer) FinishStream(finishReason string) error {
	s.mu.Lock()
	if s.hasFinished {
		s.mu.Unlock()
		return nil
	}
	s.hasFinished = true
	s.mu.Unlock()

	if err := s.EnsureMessageStart(); err != nil {
		return err
	}
	if err := s.closeCurrentBlock(); err != nil {
		return err
	}

	// A mid-generation safety cutoff with nothing produced yet is the same
	// silent-failure shape as the promptFeedback block above, just arriving
	// as a candidate's finishReason instead of a top-level field. Anthropic
	// has a real stop_reason for exactly this ("refusal") — ending the turn
	// cleanly with it is friendlier to a client than aborting the whole
	// stream with event:error, since the client still gets a complete,
	// well-formed message it can display ("the model declined to answer")
	// instead of a hard stream failure. If some content DID stream before
	// the cutoff, let it through as end_turn: the client has something to
	// show, which beats discarding it.
	isSafetyBlock := finishReason == "SAFETY" || finishReason == "RECITATION" || finishReason == "BLOCKLIST" ||
		finishReason == "PROHIBITED_CONTENT" || finishReason == "SPII"
	isEmptyRefusal := isSafetyBlock && !s.toolUseOccurred && s.textChars == 0 && s.thinkingChars == 0

	// MAX_TOKENS must win over tool_use: toolUseOccurred only means a
	// functionCall part was seen, not that it arrived complete. If
	// generation was cut off mid-arguments by the output ceiling, the
	// client needs to know the JSON it's about to parse may be truncated —
	// reporting "tool_use" here hides that and the client sees a plain
	// JSON syntax error instead of a clear "hit the length limit".
	stopReason := "end_turn"
	if isEmptyRefusal {
		stopReason = "refusal"
		// Anthropic's Message.content has minItems:1 — a refusal with
		// NOTHING ever streamed (no text_delta/thinking_delta/tool_use)
		// would otherwise leave the client reconstructing a genuinely
		// empty content array from zero content_block_start/stop events,
		// which a strict SDK's schema validation rejects outright.
		if err := s.HandleText("I'm not able to help with that request."); err != nil {
			return err
		}
		if err := s.closeCurrentBlock(); err != nil {
			return err
		}
	} else if finishReason == "MAX_TOKENS" {
		stopReason = "max_tokens"
	} else if s.toolUseOccurred {
		stopReason = "tool_use"
	}

	// Gemini's promptTokenCount is the FULL input (cached + uncached), but
	// Anthropic's input_tokens means only the uncached delta — cached reads
	// are reported separately via cache_read_input_tokens. Sending the full
	// count in both fields double-counts the cached portion: a client
	// computing total = input_tokens + cache_read_input_tokens (per
	// Anthropic's own documented formula) sees the cache tokens twice, and
	// the displayed cache-hit-rate comes out roughly half of the real one.
	// If the stream ended (e.g. a dropped connection) without Gemini ever
	// sending a final usageMetadata chunk, s.inputTokens stays at its zero
	// value — falling back to 0 here would contradict the non-zero
	// estInputTokens estimate message_start already told the client.
	if s.inputTokens == 0 && s.estInputTokens > 0 {
		s.inputTokens = int64(s.estInputTokens)
	}
	uncachedInputTokens := s.inputTokens - s.cacheTokens
	if uncachedInputTokens < 0 {
		uncachedInputTokens = 0
	}
	// Gemini never reports a distinct "tokens newly written to cache"
	// counter — only cachedContentTokenCount (tokens actually READ from an
	// existing cache). On a cache miss (nothing read yet) with a prompt
	// large enough that Google's implicit caching will cache it going
	// forward, approximate cache_creation_input_tokens as the uncached
	// amount; without this, Claude Code saw 0% caching on a session's
	// first turn even though Google silently caches it for every turn
	// after.
	var cacheCreationTokens int64
	if s.cacheTokens == 0 && uncachedInputTokens > 1024 {
		cacheCreationTokens = uncachedInputTokens
		uncachedInputTokens = 0
	}
	usage := map[string]interface{}{
		"input_tokens":  uncachedInputTokens,
		"output_tokens": s.outputTokens,
	}
	if cacheCreationTokens > 0 {
		usage["cache_creation_input_tokens"] = cacheCreationTokens
	}
	if s.cacheTokens > 0 {
		usage["cache_read_input_tokens"] = s.cacheTokens
	}

	deltaMap := map[string]interface{}{
		"stop_reason":   stopReason,
		"stop_sequence": nil,
	}

	if s.requestedSafeguards {
		toolUsesMap := make(map[string]interface{})
		for _, tid := range s.toolCallIDs {
			toolUsesMap[tid] = map[string]interface{}{
				"type":    "evaluated",
				"outcome": "not_flagged",
			}
		}
		deltaMap["safeguard_results"] = []interface{}{
			map[string]interface{}{
				"type": "dangerous_tool_use",
				"status": map[string]interface{}{
					"type":      "available",
					"tool_uses": toolUsesMap,
				},
			},
		}
	}

	deltaPayload := map[string]interface{}{
		"type":  "message_delta",
		"delta": deltaMap,
		"usage": usage,
	}
	b, _ := json.Marshal(deltaPayload)
	if err := s.SendRaw("message_delta", string(b)); err != nil {
		return err
	}

	return s.SendRaw("message_stop", `{"type":"message_stop"}`)
}

// ProcessGeminiChunk parses a single Gemini SSE JSON line and drives the state machine.
func (s *SSEStreamer) ProcessGeminiChunk(chunkJSON []byte) error {
	root := gjson.ParseBytes(chunkJSON)
	if respNode := root.Get("response"); respNode.Exists() {
		root = respNode
	}

	// Google can send an in-band error chunk mid-stream (e.g. a 429
	// RESOURCE_EXHAUSTED that only surfaces once generation is already
	// underway) and then just close the connection. That chunk has no
	// "candidates" node, so without this check the code fell straight
	// through to the candidates-empty early return, the read loop then hit
	// a normal EOF, and FinishStream("STOP") reported a clean "end_turn" —
	// the client saw a successful, if truncated, response instead of
	// learning the generation actually failed.
	if errNode := root.Get("error"); errNode.Exists() {
		msg := errNode.Get("message").String()
		if msg == "" {
			msg = errNode.Raw
		}
		// Google's in-band error code mirrors HTTP status semantics (429,
		// 503, etc.) even though the outer HTTP response was 200 OK — map
		// it the same way as a real HTTP failure so the client's SDK knows
		// whether this is retryable instead of seeing a generic api_error
		// for everything.
		errType := anthropicErrorTypeForStatus(int(errNode.Get("code").Int()))
		errPayload := fmt.Sprintf(`{"type":"error","error":{"type":%q,"message":%q}}`, errType, "Upstream stream error: "+msg)
		_ = s.SendRaw("error", errPayload)
		return fmt.Errorf("upstream stream error: %s", msg)
	}

	// A prompt Google's safety filters reject outright (before any
	// candidate is even generated) comes back as a normal 200 OK with a
	// promptFeedback node and no "candidates" or "error" field at all — the
	// same silent-swallow failure mode as the error check above, just under
	// a different key. Without this it fell straight through to the
	// candidates-empty return below and the client saw a clean, empty
	// "end_turn" instead of learning the prompt was blocked.
	if pf := root.Get("promptFeedback"); pf.Exists() {
		if br := pf.Get("blockReason").String(); br != "" && br != "BLOCK_REASON_UNSPECIFIED" {
			msg := "Google safety filters blocked this prompt: " + br
			errPayload := fmt.Sprintf(`{"type":"error","error":{"type":"api_error","message":%q}}`, msg)
			_ = s.SendRaw("error", errPayload)
			return fmt.Errorf("prompt blocked by Google safety filters: %s", br)
		}
	}

	// Update token usage
	if um := root.Get("usageMetadata"); um.Exists() {
		if pt := um.Get("promptTokenCount").Int(); pt > 0 {
			s.inputTokens = pt
		}
		if ct := um.Get("candidatesTokenCount").Int(); ct > 0 {
			s.outputTokens = ct
		}
		if cc := um.Get("cachedContentTokenCount").Int(); cc > 0 {
			s.cacheTokens = cc
		}
	}

	candidates := root.Get("candidates")
	if !candidates.IsArray() || len(candidates.Array()) == 0 {
		return nil
	}
	cand := candidates.Array()[0]

	parts := cand.Get("content.parts")
	if parts.IsArray() {
		for _, part := range parts.Array() {
			if part.Get("thought").Bool() {
				// The signature is often carried on a trailing chunk whose
				// text is empty ("") — Google's own docs warn that filtering
				// on an empty chunk.text silently drops it. Capture it
				// either way; only emit a thinking_delta when there's text.
				sig := part.Get("thoughtSignature").String()
				if t := part.Get("text").String(); t != "" {
					if err := s.HandleThinking(t, sig); err != nil {
						return err
					}
				} else if sig != "" {
					if err := s.EnsureMessageStart(); err != nil {
						return err
					}
					if s.currentType == TypeThinking {
						s.thoughtSignature = sig
					}
				}
			} else if fc := part.Get("functionCall"); fc.Exists() {
				name := fc.Get("name").String()
				args := fc.Get("args").Raw
				if args == "" {
					args = "{}"
				}
				if err := s.HandleFunctionCall(name, args, part.Get("thoughtSignature").String()); err != nil {
					return err
				}
			} else if t := part.Get("text").String(); t != "" {
				if err := s.HandleText(t); err != nil {
					return err
				}
			}
		}
	}

	// Process Grounding Metadata from Google Search
	if gm := cand.Get("groundingMetadata"); gm.Exists() {
		if queries := gm.Get("webSearchQueries"); queries.IsArray() {
			for _, q := range queries.Array() {
				s.searchQueries = append(s.searchQueries, q.String())
			}
		}
		if chunks := gm.Get("groundingChunks"); chunks.IsArray() {
			for _, c := range chunks.Array() {
				uri := c.Get("web.uri").String()
				title := c.Get("web.title").String()
				if uri != "" {
					if title == "" {
						title = uri
					}
					s.groundingSources = append(s.groundingSources, GroundingSource{
						Title: title,
						URI:   uri,
					})
				}
			}
		}
	}

	if fr := cand.Get("finishReason").String(); fr != "" && fr != "FINISH_REASON_UNSPECIFIED" {
		// Append Grounding Sources footer if Google Search was performed —
		// but never after a tool_use block: per Anthropic convention a
		// tool_use block ends the assistant's turn (the client replies with
		// its tool_result next), and trailing text after it risks confusing
		// stricter SDK parsers.
		if len(s.groundingSources) > 0 && !s.toolUseOccurred {
			var sb strings.Builder
			if s.preferRussian {
				sb.WriteString("\n\n---\n**Источники Google Search:**\n")
			} else {
				sb.WriteString("\n\n---\n**Google Search Sources:**\n")
			}
			seen := make(map[string]bool)
			for _, src := range s.groundingSources {
				if !seen[src.URI] {
					seen[src.URI] = true
					sb.WriteString(fmt.Sprintf("- [%s](%s)\n", src.Title, src.URI))
				}
			}
			_ = s.HandleText(sb.String())
		}
		return s.FinishStream(fr)
	}

	return nil
}

// StreamGeminiToClaude reads lines from Gemini stream body and translates to
// Claude SSE events. onActivity, when non-nil, is called every time bytes
// are actually read from upstream — the caller uses this to run an
// idle-timeout watchdog that only cancels on genuine silence, not on total
// elapsed duration (a single buffered function-call chunk for a huge Write
// can legitimately take minutes to arrive).
func StreamGeminiToClaude(streamer *SSEStreamer, r io.Reader, onActivity func()) error {
	// Start 5-second keepalive heartbeat worker to prevent Claude Code idle timeouts during pauses
	pingTicker := time.NewTicker(5 * time.Second)
	defer pingTicker.Stop()

	stopPing := make(chan struct{})
	defer close(stopPing)

	go func() {
		for {
			select {
			case <-pingTicker.C:
				_ = streamer.SendKeepalivePing()
			case <-stopPing:
				return
			}
		}
	}()

	buf := make([]byte, 64*1024)
	var leftOver string

	for {
		n, err := r.Read(buf)
		if n > 0 {
			if onActivity != nil {
				onActivity()
			}
			data := leftOver + string(buf[:n])
			lines := strings.Split(data, "\n")
			leftOver = lines[len(lines)-1]

			for i := 0; i < len(lines)-1; i++ {
				line := strings.TrimSpace(lines[i])
				if strings.HasPrefix(line, "data:") {
					payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
					if payload == "[DONE]" {
						return streamer.FinishStream("STOP")
					}
					if strings.HasPrefix(payload, "{") {
						if err := streamer.ProcessGeminiChunk([]byte(payload)); err != nil {
							return err
						}
					}
				}
			}
		}
		if err != nil {
			if err == io.EOF {
				break
			}
			return err
		}
	}

	if strings.TrimSpace(leftOver) != "" {
		line := strings.TrimSpace(leftOver)
		if strings.HasPrefix(line, "data:") {
			payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if strings.HasPrefix(payload, "{") {
				_ = streamer.ProcessGeminiChunk([]byte(payload))
			}
		}
	}

	return streamer.FinishStream("STOP")
}
