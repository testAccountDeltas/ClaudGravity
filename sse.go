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
			sweepStaleSessionCacheFractions()
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

// sessionCacheFraction remembers, per session (keyed the same way cwdKey/
// loopKey already are), the last REAL cache_read/input_tokens ratio this
// conversation saw at the end of a turn. message_start has to guess its
// usage numbers before Gemini has even started generating — Google's real
// cache split only arrives in the stream's last chunk — so it always sent
// cache_read_input_tokens:0 there. Official Anthropic SDKs latch
// input_tokens/cache_read_input_tokens from message_start only and never
// update them from message_delta's real values, so every client built on
// one of those SDKs displayed a permanent 0% cache hit rate regardless of
// the real (correct) numbers later turns' message_delta carried. Guessing
// from the PREVIOUS turn's real ratio is the best available estimate for
// a conversation that's actively warm — still corrected to the real
// number in message_delta for anything that reads that instead.
var (
	sessionCacheFractionMu   sync.Mutex
	sessionCacheFractionMap  = make(map[string]float64)
	sessionCacheFractionSeen = make(map[string]time.Time)
)

func sweepStaleSessionCacheFractions() {
	sessionCacheFractionMu.Lock()
	defer sessionCacheFractionMu.Unlock()
	cutoff := time.Now().Add(-toolSignatureStaleAfter)
	for id, t := range sessionCacheFractionSeen {
		if t.Before(cutoff) {
			delete(sessionCacheFractionSeen, id)
			delete(sessionCacheFractionMap, id)
		}
	}
}

func setSessionCacheFraction(sessionID string, frac float64) {
	if sessionID == "" {
		return
	}
	sessionCacheFractionMu.Lock()
	defer sessionCacheFractionMu.Unlock()
	sessionCacheFractionMap[sessionID] = frac
	sessionCacheFractionSeen[sessionID] = time.Now()
}

func getSessionCacheFraction(sessionID string) (float64, bool) {
	sessionCacheFractionMu.Lock()
	defer sessionCacheFractionMu.Unlock()
	f, ok := sessionCacheFractionMap[sessionID]
	return f, ok
}

type SSEStreamer struct {
	rid                      uint64 // correlates with globalSpecCapture's per-request record
	w                        http.ResponseWriter
	flusher                  http.Flusher
	mu                       sync.Mutex
	hasStarted               bool
	currentType              int // 0: none, 1: text, 2: thinking, 3: tool_use
	currentIndex             int
	messageID                string
	model                    string
	inputTokens              int64
	outputTokens             int64
	cacheTokens              int64
	toolUseOccurred          bool
	textChars                int
	thinkingChars            int
	toolNames                []string
	toolCallSignatures       []string
	lastText                 string
	lastEventTime            time.Time
	searchQueries            []string
	groundingSources         []GroundingSource
	sessionID                string
	requestedSafeguards      bool
	toolCallIDs              []string
	hasFinished              bool
	thoughtSignature         string          // most recent thoughtSignature seen for the open thinking block, flushed as signature_delta on close
	lastTurnThoughtSignature string          // durable across the turn; inherited by function calls and parallel tool calls
	thinkingFinished         bool            // true once a thinking block has been opened and closed — Anthropic allows at most one, so a later "thought" chunk (Gemini can interleave thought->text->thought) must not reopen a second one
	hasPriorTurns            bool            // whether request had assistant messages in history (prior turns in the conversation)
	estInputTokens           int             // approximate input token count, known upfront unlike Gemini's real usageMetadata (only arrives at stream end)
	preferRussian            bool            // conversation's own language, detected from the user's latest message — picks which language gateway-synthesized text (grounding footer) uses, since that text bypasses the model entirely
	syntheticReasonTools     map[string]bool // tool names whose Gemini-bound schema got our synthetic "reason" placeholder injected (translator.go); that key is stripped back out of the model's actual function-call args before handing them to Claude Code
	cachePrefixFrac          float64         // 0 if the client's request set no Anthropic cache_control breakpoint; otherwise the approximate fraction of uncached input tokens that sit before the breakpoint — gates AND scales the cache_creation_input_tokens simulation in FinishStream, see cacheBreakpointPrefixFraction's doc comment
	disableParallelToolUse   bool            // client's disable_parallel_tool_use — HandleFunctionCall drops any call past the first when set, see wantsSingleToolUse's doc comment
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

func NewSSEStreamer(rid uint64, w http.ResponseWriter, flusher http.Flusher, model string, sessionID string, requestedSafeguards bool, estInputTokens int, preferRussian bool, syntheticReasonTools map[string]bool, hasPriorTurns bool, cachePrefixFrac float64, disableParallelToolUse bool) *SSEStreamer {
	return &SSEStreamer{
		rid:                    rid,
		w:                      w,
		flusher:                flusher,
		messageID:              fmt.Sprintf("msg_%d", time.Now().UnixNano()),
		model:                  model,
		currentIndex:           -1,
		lastEventTime:          time.Now(),
		sessionID:              sessionID,
		requestedSafeguards:    requestedSafeguards,
		estInputTokens:         estInputTokens,
		preferRussian:          preferRussian,
		syntheticReasonTools:   syntheticReasonTools,
		cachePrefixFrac:        cachePrefixFrac,
		disableParallelToolUse: disableParallelToolUse,
		hasPriorTurns:          hasPriorTurns,
	}
}

func (s *SSEStreamer) SendRaw(event, data string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastEventTime = time.Now()
	globalSpecCapture.AppendClaudeEvent(s.rid, event, data)
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
	// Gemini's own usageMetadata (real cache split) only arrives in the
	// stream's LAST chunk, so the true split can't be known this early —
	// but official Anthropic SDKs latch input_tokens/cache_read_input_tokens
	// from message_start ONLY and never revise them from message_delta's
	// real numbers later, so always sending 0 here left every one of those
	// clients permanently displaying 0% cache hit regardless of the correct
	// numbers message_delta carried at the end. Guess from this same
	// session's last turn (an actively warm conversation's cache fraction
	// doesn't usually swing much turn to turn) when one is on record;
	// message_delta still corrects to the real number either way.
	cachedEstimate := int64(0)
	uncachedEstimate := int64(s.estInputTokens)
	// Only estimate cache read if prompt is large enough to be cache-eligible (>1024 tokens)
	// and the conversation has prior assistant turns. A brand new prompt or cold turn
	// must not inherit a false 85% cache hit.
	if s.estInputTokens >= 1024 && s.hasPriorTurns {
		if frac, ok := getSessionCacheFraction(s.sessionID); ok && frac > 0 {
			cachedEstimate = int64(float64(s.estInputTokens) * frac)
			uncachedEstimate = int64(s.estInputTokens) - cachedEstimate
			if uncachedEstimate < 0 {
				uncachedEstimate = 0
			}
		}
	}
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
			"usage": map[string]interface{}{
				"input_tokens":                uncachedEstimate,
				"output_tokens":               0,
				"cache_creation_input_tokens": 0,
				"cache_read_input_tokens":     cachedEstimate,
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
		s.thinkingFinished = true
		sig := s.thoughtSignature
		if sig == "" {
			sig = s.lastTurnThoughtSignature
		}
		if sig == "" {
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
	if (s.thinkingFinished || s.currentIndex >= 0) && s.currentType != TypeThinking {
		// Gemini can interleave thought -> text -> thought within one
		// candidate; Anthropic allows at most one "thinking" block and it
		// must be first. A second one a strict SDK's schema validation
		// rejects outright, so once the first has closed (or any non-thinking block
		// has opened), fold any further "thought" chunks into plain text instead of reopening one.
		return s.HandleText(text)
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
		s.lastTurnThoughtSignature = thoughtSignature
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
	if s.disableParallelToolUse && s.toolUseOccurred {
		// Client asked for at most one tool call per turn. The system
		// instruction added at translation time (translator.go) is only a
		// request Gemini can ignore — this is the actual enforcement:
		// silently drop any call past the first rather than streaming a
		// second tool_use block the client explicitly said it can't handle.
		return nil
	}
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
	effectiveSig := thoughtSignature
	if effectiveSig == "" && s.currentType == TypeThinking {
		effectiveSig = s.thoughtSignature
	}
	if effectiveSig == "" {
		effectiveSig = s.lastTurnThoughtSignature
	}
	if s.currentType == TypeThinking && s.thoughtSignature == "" && effectiveSig != "" {
		s.thoughtSignature = effectiveSig
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
	setToolSignature(callID, effectiveSig)

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
	// event:error is itself terminal in Anthropic's SSE state machine — the
	// other two event:error sites in this file (ProcessGeminiChunk's
	// upstream-error and blocked-prompt paths) never follow it with
	// message_stop either, and a strict client parser that expects nothing
	// more after "error" would be confused by one arriving anyway.
	errPayload := fmt.Sprintf(`{"type":"error","error":{"type":"api_error","message":%q}}`, "Stream connection broken: "+err.Error())
	_ = s.SendRaw("error", errPayload)
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
	// instead of a hard stream failure.
	isSafetyBlock := finishReason == "SAFETY" || finishReason == "RECITATION" || finishReason == "BLOCKLIST" ||
		finishReason == "PROHIBITED_CONTENT" || finishReason == "SPII"

	stopReason := "end_turn"
	if isSafetyBlock {
		stopReason = "refusal"
		if s.currentIndex < 0 {
			if err := s.HandleText("I'm not able to help with that request."); err != nil {
				return err
			}
			if err := s.closeCurrentBlock(); err != nil {
				return err
			}
		}
	} else if s.currentIndex < 0 {
		// Anthropic Messages API requires at least one content block (minItems: 1).
		// If generation finished without emitting any thinking, text, or tool_use block,
		// emit a fallback block. If a thinking block or tool call was already streamed,
		// minItems: 1 is already satisfied and we must NEVER inject synthetic dots ("...")
		// which would be visible in Claude Code's terminal output.
		if err := s.HandleText("(empty response)"); err != nil {
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
	// are reported separately via cache_read_input_tokens.
	if s.inputTokens == 0 && s.estInputTokens > 0 {
		s.inputTokens = int64(s.estInputTokens)
	}
	uncachedInputTokens := s.inputTokens - s.cacheTokens
	if uncachedInputTokens < 0 {
		uncachedInputTokens = 0
	}

	var cacheCreationTokens int64
	// Gated on the request actually having set a cache_control breakpoint
	// — an audit caught this firing on every large, cache-miss prompt
	// regardless of whether caching was ever requested, misreporting a
	// plain request as one that paid for a (more expensive) cache write.
	//
	// A second audit caught that treating the ENTIRE uncached amount as
	// cache_creation (zeroing input_tokens outright) overstates it: only
	// the prefix up to the breakpoint is actually written to cache, and
	// the current turn's own new content after it should stay counted as
	// plain input_tokens. s.cachePrefixFrac approximates that split.
	if s.cacheTokens == 0 && uncachedInputTokens > cacheCreationThreshold(s.model) && s.cachePrefixFrac > 0 {
		cacheCreationTokens = int64(float64(uncachedInputTokens) * s.cachePrefixFrac)
		// In Anthropic's formula: Total prompt tokens = input_tokens + cache_creation_input_tokens + cache_read_input_tokens.
		// Tokens that created cache must NOT be counted in input_tokens as well, or SDKs double-count every token.
		uncachedInputTokens -= cacheCreationTokens
	}

	// Record this turn's real cache fraction for the NEXT turn's message_start estimate
	if s.inputTokens > 0 {
		if s.cacheTokens > 0 {
			setSessionCacheFraction(s.sessionID, float64(s.cacheTokens)/float64(s.inputTokens))
		} else if cacheCreationTokens > 0 {
			// Turn 1 created cache for this prompt. On Turn 2, the vast majority
			// of the prompt will be a cache hit. Seed with 0.85 so message_start on Turn 2
			// accurately reflects prompt caching instead of latching 0%.
			setSessionCacheFraction(s.sessionID, 0.85)
		}
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
	if stopReason == "refusal" {
		deltaMap["stop_details"] = map[string]interface{}{
			"type":        "refusal",
			"category":    "general_harms",
			"explanation": "Content blocked by safety policy",
		}
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
	globalSpecCapture.AppendGeminiChunk(s.rid, chunkJSON)
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
		// for everything. The google.rpc.Status "status" string (e.g.
		// "RESOURCE_EXHAUSTED", "UNAVAILABLE") is unambiguous regardless of
		// whether "code" on this particular envelope turns out to be the
		// HTTP-style number the comment above assumes or a raw gRPC status
		// code (8, 14, ...) instead — checked first as a safe supplement,
		// never a behavior change for whatever "code" already handles
		// correctly.
		var errType string
		switch errNode.Get("status").String() {
		case "RESOURCE_EXHAUSTED":
			errType = "rate_limit_error"
		case "UNAVAILABLE":
			errType = "overloaded_error"
		case "INVALID_ARGUMENT", "FAILED_PRECONDITION", "OUT_OF_RANGE":
			errType = "invalid_request_error"
		case "UNAUTHENTICATED":
			errType = "authentication_error"
		case "PERMISSION_DENIED":
			errType = "permission_error"
		case "NOT_FOUND":
			errType = "not_found_error"
		default:
			errType = anthropicErrorTypeForStatus(int(errNode.Get("code").Int()))
		}
		errPayload := fmt.Sprintf(`{"type":"error","error":{"type":%q,"message":%q}}`, errType, "Upstream stream error: "+msg)
		_ = s.SendRaw("error", errPayload)
		s.mu.Lock()
		s.hasFinished = true
		s.mu.Unlock()
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
			_ = s.EnsureMessageStart()
			_ = s.HandleText("I am unable to complete this request because it was flagged by upstream safety filters: " + br)
			_ = s.closeCurrentBlock()
			return s.FinishStream("SAFETY")
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
			partSig := part.Get("thoughtSignature").String()
			if partSig != "" {
				s.lastTurnThoughtSignature = partSig
				if s.thoughtSignature == "" {
					s.thoughtSignature = partSig
				}
			}
			if part.Get("thought").Bool() {
				// The signature is often carried on a trailing chunk whose
				// text is empty ("") — Google's own docs warn that filtering
				// on an empty chunk.text silently drops it. Capture it
				// either way; only emit a thinking_delta when there's text.
				sig := partSig
				if t := part.Get("text").String(); t != "" {
					if err := s.HandleThinking(t, sig); err != nil {
						return err
					}
				} else if sig != "" {
					if err := s.EnsureMessageStart(); err != nil {
						return err
					}
					s.lastTurnThoughtSignature = sig
					if s.currentType == TypeThinking {
						s.thoughtSignature = sig
					}
				}
			} else if fc := part.Get("functionCall"); fc.Exists() {
				name := fc.Get("name").String()
				args := fc.Get("args").Raw
				if args == "" || args == "null" {
					// Anthropic's tool_use.input must be an object; a literal
					// JSON null here (Gemini can return bare "args": null for
					// a no-argument call) streamed through as partial_json
					// "null" instead of "{}", which Claude Code's own
					// incremental JSON parser can't treat as an object.
					args = "{}"
				}
				callSig := partSig
				if callSig == "" {
					callSig = s.lastTurnThoughtSignature
				}
				if err := s.HandleFunctionCall(name, args, callSig); err != nil {
					return err
				}
			} else if t := part.Get("text").String(); t != "" {
				if s.currentType == TypeThinking && s.thoughtSignature == "" && partSig != "" {
					s.thoughtSignature = partSig
				}
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
