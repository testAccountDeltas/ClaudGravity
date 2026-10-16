package main

import (
	"encoding/json"
	"fmt"
)

// contextSafetyMarginTokens reserves headroom beyond the reserved output
// budget, since our token estimator (token_estimator.go) is an approximation,
// not the real tokenizer Google uses.
const contextSafetyMarginTokens = 2000

// conservativeTokenEstimate deliberately ignores token_estimator.go's
// character-class weighting (optimized for realistic prose, ~4.5 chars/
// token for plain ASCII) and instead uses a flat, pessimistic 2 chars/token.
// Empirically, the weighted estimator can undercount by 2x+ on content that
// doesn't match its assumptions (verified: a 280,000-char synthetic payload
// came back as 141,065 real tokens from Google — ~2.0 chars/token — while
// the weighted estimator said ~62,500). This function exists ONLY for the
// truncation safety check below, where undercounting means a request slips
// through and hits a raw context-length 400 from Google — so it must err
// on the side of over-truncating, never under-truncating.
func conservativeTokenEstimate(v interface{}) int {
	if v == nil {
		return 0
	}
	b, err := json.Marshal(v)
	if err != nil {
		return 0
	}
	n := len(b) / 2
	if n < 1 {
		return 1
	}
	return n
}

// TruncateConversationToFit drops the oldest messages when the estimated
// total context would exceed the REAL context window of the Gemini model
// that tiering (ResolveTieringAndEffort) actually picked for this request.
//
// This exists because CLAUDE_CODE_MAX_CONTEXT_TOKENS / AUTO_COMPACT_WINDOW
// are static client-side settings Claude Code uses to decide when to
// compact its own history — but tiering can route different requests
// within the SAME session to different Gemini sub-models (low/medium/high/
// tiered), each with its own real ceiling from Google's live model catalog
// (globalModelCatalog). The client has no way to know which one will serve
// a given request, so it cannot size its own compaction threshold
// correctly; this is the gateway-side backstop so an oversized request
// gets trimmed here instead of failing upstream with a raw context-length
// error from Google.
// fallbackMaxTokens covers the known Gemini model family ceilings for the
// rare case globalModelCatalog is empty (startup FetchAvailableModels
// failed, or ran before the network was up) — without this, truncation
// silently disabled itself entirely rather than erring conservative, and a
// request would sail right past the 1M-token wall into a raw Google 400
// instead of being proactively trimmed here.
var fallbackMaxTokens = map[string]int64{
	"gemini-3.8-flash":        1048576,
	"gemini-3.8-flash-tiered": 1048576,
	"gemini-3.8-flash-low":    1048576,
	"gemini-3.8-flash-medium": 1048576,
	"gemini-3.8-flash-high":   1048576,
	"gemini-2.5-flash":        1048576,
	"gemini-2.5-pro":          2097152,
	"gemini-2.5-flash-lite":   1048576,
}

func TruncateConversationToFit(reqMap map[string]interface{}, modelName string, requestedMaxOutput int64) (droppedCount int, truncated bool) {
	modelMeta, hasMeta := globalModelCatalog.GetModel(modelName)
	maxTokens := modelMeta.MaxTokens
	if !hasMeta || maxTokens <= 0 {
		maxTokens = fallbackMaxTokens[modelName]
		if maxTokens <= 0 {
			return 0, false
		}
	}

	msgsRaw, ok := reqMap["messages"]
	if !ok {
		return 0, false
	}
	msgsList, ok := msgsRaw.([]interface{})
	if !ok || len(msgsList) < 4 {
		// Too small to meaningfully trim, and dropping from a tiny history
		// risks breaking role alternation for no benefit.
		return 0, false
	}

	budget := maxTokens - requestedMaxOutput - contextSafetyMarginTokens
	if budget <= 0 {
		budget = maxTokens / 2
	}

	systemTokens := 0
	if sys, ok := reqMap["system"]; ok {
		systemTokens = conservativeTokenEstimate(sys)
	}
	if tools, ok := reqMap["tools"]; ok {
		systemTokens += conservativeTokenEstimate(tools)
	}

	msgTokens := make([]int, len(msgsList))
	total := systemTokens
	for i, m := range msgsList {
		msgTokens[i] = conservativeTokenEstimate(m) + 3 // per-message framing overhead
		total += msgTokens[i]
	}

	if total <= int(budget) {
		return 0, false
	}

	// Drop oldest messages first. SanitizeMessagesAndGraph runs right after
	// this in ConvertClaudeRequestToGemini and repairs any orphaned
	// tool_use/tool_result pairing or role-alternation issue this leaves
	// behind, so a simple prefix-drop here is sufficient.
	dropped := 0
	i := 0
	for total > int(budget) && i < len(msgsList)-2 {
		total -= msgTokens[i]
		dropped++
		i++
	}
	if dropped == 0 {
		return 0, false
	}

	remaining := msgsList[i:]
	notice := map[string]interface{}{
		"role": "user",
		"content": fmt.Sprintf(
			"[Gateway notice: %d earlier message(s) were dropped from this conversation to fit inside the %s model's %d-token context window. The conversation continues below.]",
			dropped, modelName, maxTokens,
		),
	}
	reqMap["messages"] = append([]interface{}{notice}, remaining...)
	return dropped, true
}
