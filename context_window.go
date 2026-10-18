package main

import (
	"encoding/json"
	"strings"
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
	// A message's "content" array can hold image/document blocks whose
	// base64 payload is pure byte bulk with none of the 2-chars/token
	// density the flat-rate fallback below assumes — a single 3MB
	// screenshot (~4,000,000 base64 chars) would price out at ~2,000,000
	// "tokens" even though Gemini bills it ~258-1600 tokens, which made the
	// truncation loop below mistake one attached image for a context
	// overflow and drop the entire rest of the conversation to compensate.
	if m, ok := v.(map[string]interface{}); ok {
		if blocks, isList := m["content"].([]interface{}); isList {
			total := 0
			counted := false
			for _, b := range blocks {
				bMap, isMap := b.(map[string]interface{})
				if !isMap {
					continue
				}
				total += blockTokenEstimate(bMap)
				counted = true
			}
			if counted {
				return total
			}
		}
	}
	return flatTokenEstimate(v)
}

// blockTokenEstimate prices a single content block, recursing into
// "tool_result" blocks whose own "content" is itself a sub-array of blocks
// (Anthropic's tool_result can carry image/document results from the
// tool, e.g. a browser screenshot or a Read of a PDF). Without this, a
// screenshot sitting inside a tool_result fell through to the same
// len(json)/2 flat estimate conservativeTokenEstimate exists to avoid for
// a top-level image block — only one level shallower.
func blockTokenEstimate(bMap map[string]interface{}) int {
	switch bMap["type"] {
	case "image":
		return 1600
	case "document":
		// If it's a base64 PDF, scale with realistic PDF token density (~258 tokens/page,
		// ~1 token per 500 base64 chars) with a floor of 2500 tokens.
		// Using flatTokenEstimate (len/2) treats base64 text as raw ASCII tokens, pricing a 2MB
		// PDF at 1.33M tokens and instantly destroying conversation history.
		if src, ok := bMap["source"].(map[string]interface{}); ok {
			if dataStr, isStr := src["data"].(string); isStr {
				if src["type"] == "base64" {
					// An audit caught that fmt.Sprint on a missing
					// media_type key (a nil interface{}) returns the
					// literal string "<nil>", not "" — so the intended
					// "unknown type defaults to PDF pricing" fallback
					// below never actually triggered; a PDF with no
					// media_type fell through to the raw-text-density
					// formula instead, which treats base64's high-entropy
					// bytes as if decoded text, inflating a few-thousand-
					// token PDF into a six-figure estimate and needlessly
					// evicting history to "fit" a budget it was never
					// close to. Extract the string directly so a missing
					// or non-string media_type yields "" as intended.
					var mediaType string
					if mt, ok := src["media_type"].(string); ok {
						mediaType = strings.ToLower(mt)
					}
					if mediaType == "application/pdf" || mediaType == "" {
						est := len(dataStr) / 500
						if est < 2500 {
							return 2500
						}
						return est
					}
					// For base64-encoded text documents (code, logs, CSVs), decode to raw text density:
					decodedBytes := len(dataStr) * 3 / 4
					est := decodedBytes / 2
					if est < 1 {
						return 1
					}
					return est
				}
				// Plain text document
				return flatTokenEstimate(dataStr)
			}
		}
		return 2500
	case "tool_result":
		inner, isList := bMap["content"].([]interface{})
		if !isList {
			return flatTokenEstimate(bMap)
		}
		total := 0
		for _, b := range inner {
			if innerMap, isMap := b.(map[string]interface{}); isMap {
				total += blockTokenEstimate(innerMap)
			}
		}
		return total
	default:
		return flatTokenEstimate(bMap)
	}
}

func flatTokenEstimate(v interface{}) int {
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
			// An audit caught that a model name present in neither the
			// (possibly empty, if FetchAvailableModels failed/raced at
			// startup) live catalog NOR this hardcoded fallback map —
			// a new Gemini release, or a custom/renamed model — silently
			// disabled truncation entirely instead of erring conservative,
			// the exact failure case fallbackMaxTokens itself exists to
			// prevent. Fall back to the smallest known real ceiling rather
			// than giving up: trimming a request that didn't need it is
			// harmless; sailing an oversized one straight into Google's
			// raw context-length 400 is not.
			maxTokens = 1048576
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

	// Drop oldest INTERMEDIATE messages first, starting at index 1: index 0
	// is the user's original task statement for this conversation, and
	// erasing it left the agent with no idea what it was asked to do once
	// enough history piled up to trigger truncation — it would start asking
	// the user to restate the goal mid-task. SanitizeMessagesAndGraph runs
	// right after this in ConvertClaudeRequestToGemini and repairs any
	// orphaned tool_use/tool_result pairing or role-alternation issue this
	// leaves behind, so a simple prefix-drop here is sufficient.
	dropped := 0
	i := 1
	for total > int(budget) && i < len(msgsList)-2 {
		total -= msgTokens[i]
		dropped++
		i++
	}
	if dropped == 0 {
		return 0, false
	}

	remaining := msgsList[i:]
	// Static text, no dropped-count/model-name/max-tokens interpolation:
	// this notice sits near the very start of the conversation, so once
	// truncation first fires, EVERY byte after it is part of Google's
	// server-side prefix-cache key for the rest of the conversation. The
	// dropped count grows monotonically every time this fires again on a
	// long conversation (5 -> 8 -> 12 ...), and a changed byte here
	// invalidated that prefix cache for the entire remaining history on
	// every single later turn instead of just the first truncation.
	notice := map[string]interface{}{
		"role":    "user",
		"content": "[Gateway notice: Earlier intermediate messages were dropped to fit inside this conversation's context window. The conversation continues below.]",
	}
	reqMap["messages"] = append([]interface{}{msgsList[0], notice}, remaining...)
	return dropped, true
}
