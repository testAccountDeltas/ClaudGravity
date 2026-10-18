package main

import (
	"fmt"
	"regexp"
	"strings"
	"unicode/utf8"
)

const ToolOutputCharLimit = 65536 // 64 KB limit

// extractPlainText concatenates the text of every "text" content block in a
// message content value — either already-normalized ([]map[string]interface{},
// built by this file's own Step 1 loop) or raw-from-json.Unmarshal
// ([]interface{} — a plain json.Unmarshal into interface{} never produces
// []map[string]interface{} for a JSON array, only []interface{} with each
// element itself a map[string]interface{}; this used to only handle the
// normalized shape, so calling it on a RAW message — e.g. messages[0] before
// Step 1 even runs — silently returned "" every time), plus a bare string
// content value.
func extractPlainText(content interface{}) string {
	var sb strings.Builder
	switch c := content.(type) {
	case string:
		return c
	case []map[string]interface{}:
		for _, b := range c {
			if b["type"] == "text" {
				if t, ok := b["text"].(string); ok {
					if sb.Len() > 0 && !strings.HasSuffix(sb.String(), "\n") {
						sb.WriteString("\n")
					}
					sb.WriteString(t)
				}
			}
		}
	case []interface{}:
		for _, item := range c {
			b, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			if b["type"] == "text" {
				if t, ok := b["text"].(string); ok {
					if sb.Len() > 0 && !strings.HasSuffix(sb.String(), "\n") {
						sb.WriteString("\n")
					}
					sb.WriteString(t)
				}
			}
		}
	}
	return sb.String()
}

var (
	pollutedToolCallTextPattern = regexp.MustCompile(`(?i)\[Called tool [^\]]*\]`)
	multiNewlinePattern         = regexp.MustCompile(`\n{3,}`)
)

// toolResultPlainText extracts the text out of a tool_result "content"
// value, which can be either a bare string or an array of content blocks
// (text/image/document — only "text" blocks contribute here).
func toolResultPlainText(content interface{}) string {
	switch c := content.(type) {
	case string:
		return c
	case []interface{}:
		var sb strings.Builder
		for _, item := range c {
			block, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			if block["type"] == "text" {
				if t, ok := block["text"].(string); ok {
					sb.WriteString(t)
				}
			}
		}
		return sb.String()
	}
	return ""
}

// firstNonEmptyFromMap returns the first non-empty string value found in m
// among the given keys, in order.
func firstNonEmptyFromMap(m map[string]interface{}, keys ...string) string {
	for _, k := range keys {
		if v, ok := m[k].(string); ok && v != "" {
			return v
		}
	}
	return ""
}

// dedupedRepeatedReads collapses any Read tool_result whose content is
// byte-identical to an earlier Read of the same (canonicalized) file path
// elsewhere in this same conversation into a short marker, leaving the
// first occurrence untouched. Returns true if anything was changed. A
// change in content (the file genuinely changed between reads) is left
// alone — only exact repeats are folded.
func dedupedRepeatedReads(newMsgs []map[string]interface{}) bool {
	toolIDToPath := make(map[string]string)
	seenContent := make(map[string]string)
	changed := false

	for _, m := range newMsgs {
		blocks, ok := m["content"].([]map[string]interface{})
		if !ok {
			continue
		}
		for _, b := range blocks {
			switch b["type"] {
			case "tool_use":
				name, _ := b["name"].(string)
				if !strings.EqualFold(name, "Read") {
					continue
				}
				id, _ := b["id"].(string)
				input, _ := b["input"].(map[string]interface{})
				if id == "" || input == nil {
					continue
				}
				if path := firstNonEmptyFromMap(input, "file_path", "path", "target_file"); path != "" {
					key := canonicalSnapshotKey(path)
					if offset := input["offset"]; offset != nil {
						key = fmt.Sprintf("%s:off=%v", key, offset)
					}
					if limit := input["limit"]; limit != nil {
						key = fmt.Sprintf("%s:lim=%v", key, limit)
					}
					if pages := input["pages"]; pages != nil {
						key = fmt.Sprintf("%s:pages=%v", key, pages)
					}
					toolIDToPath[id] = key
				}
			case "tool_result":
				tid, _ := b["tool_use_id"].(string)
				path, ok := toolIDToPath[tid]
				if !ok {
					continue
				}
				text := toolResultPlainText(b["content"])
				if text == "" {
					continue
				}
				// An audit caught that on a SECOND pass over the same
				// messages (SanitizeMessagesAndGraph can run more than
				// once per request), a block this function already
				// marked on a prior pass now holds the marker text
				// itself, not the real file content — reading it back in
				// here and caching it under seenContent[path] replaced
				// the real content a later genuine duplicate needs to
				// compare against, breaking dedup for any further
				// occurrence of this same path later in the pass.
				if strings.HasPrefix(text, "[Identical to the Read result already shown earlier") {
					continue
				}
				if prev, seen := seenContent[path]; seen && prev == text {
					b["content"] = fmt.Sprintf("[Identical to the Read result already shown earlier in this conversation for %s — unchanged, see above. Do not re-read; proceed with what you already have.]", path)
					changed = true
					continue
				}
				seenContent[path] = text
			}
		}
	}
	return changed
}

// StripPollutedToolCallText strips out pseudo-tool call narrations
// such as "[Called tool exec_command with input {...}]" which confuse models
// into generating tool invocations as raw text instead of structured JSON calls.
func StripPollutedToolCallText(content string) string {
	if !strings.Contains(content, "[Called tool") && !strings.Contains(content, "[called tool") {
		return content
	}
	cleaned := pollutedToolCallTextPattern.ReplaceAllString(content, "")
	cleaned = multiNewlinePattern.ReplaceAllString(cleaned, "\n\n")
	return strings.TrimSpace(cleaned)
}

// TruncateMiddle truncates large text from the center, leaving head and tail intact.
// Safely operates on UTF-8 rune boundaries to prevent invalid Unicode sequences.
// Preserves Append-Only prompt caching invariants across turns.
func TruncateMiddle(text string) (string, bool) {
	n := len(text)
	if n <= ToolOutputCharLimit {
		return text, false
	}
	// Reserve room for the marker itself (its two counts can each run to
	// ~20 digits for a pathological input) so head+marker+tail never
	// exceeds ToolOutputCharLimit — otherwise the result re-triggers the
	// n <= ToolOutputCharLimit check above on the next pass (this function
	// runs once per SanitizeMessagesAndGraph call, and that now runs more
	// than once per request) and keeps re-truncating/re-stamping a new
	// marker over the previous one every time.
	const truncateMarkerBudget = 128
	half := (ToolOutputCharLimit - truncateMarkerBudget) / 2

	// Align head cut to valid UTF-8 boundary
	headIdx := half
	for headIdx > 0 && !utf8.RuneStart(text[headIdx]) {
		headIdx--
	}

	// Align tail cut to valid UTF-8 boundary
	tailIdx := n - half
	for tailIdx < n && !utf8.RuneStart(text[tailIdx]) {
		tailIdx++
	}

	marker := fmt.Sprintf("\n<truncated %d of %d chars from the middle of the output>\n", tailIdx-headIdx, n)
	return text[:headIdx] + marker + text[tailIdx:], true
}

// blocksToInterfaceSlice converts a concrete []map[string]interface{} content
// block list into a plain []interface{} — the type every OTHER consumer of
// claudeReqJSON["messages"][i]["content"] (context_window.go's
// conservativeTokenEstimate, translator.go's lastTurnHadFailedEdit,
// loopguard.go's InjectUserNudge) type-switches on, because that's the type
// encoding/json itself produces when the original request is decoded. Any
// "content" field this file sets directly from a concrete
// []map[string]interface{} must go through this first, or those downstream
// type assertions silently fail (wrong token estimate, blind edit-failure
// detector, duplicate "user" turns — see commit history for the incident
// this caused when SanitizeMessagesAndGraph started running more than once
// per request).
func blocksToInterfaceSlice(blocks []map[string]interface{}) []interface{} {
	out := make([]interface{}, len(blocks))
	for i, b := range blocks {
		out[i] = b
	}
	return out
}

// SanitizeMessagesAndGraph performs AST-level pre-flight sanitization on conversation messages:
// 1. Eager middle-truncates tool_results (>64KB).
// 2. Removes empty/whitespace text blocks with a fallback sentinel ("...").
// 3. Ensures the first message has role "user".
// 4. Merges consecutive messages of the same role.
// 5. Converts orphaned tool_results (without tool_use) into plain text.
// 6. Synthesizes recovery responses for abandoned/unanswered tool_use calls.
// Returns (modified bool, truncatedAny bool).
func SanitizeMessagesAndGraph(claudeReqJSON map[string]interface{}) (bool, bool) {
	msgsRaw, ok := claudeReqJSON["messages"]
	if !ok {
		return false, false
	}
	msgsList, ok := msgsRaw.([]interface{})
	if !ok || len(msgsList) == 0 {
		return false, false
	}

	modified := false
	truncatedAny := false

	// Anthropic's real Messages API only allows "user"/"assistant" in
	// messages[].role — "system" belongs in the separate top-level "system"
	// field — but a client built against the OpenAI-style convention (some
	// of the non-Claude-Code web/Electron clients this gateway also serves,
	// per commonAPIMiddleware's CORS support) can still send messages[0]
	// with role "system". Step 2 below used to see that as "first message
	// isn't user" and prepend a synthetic "Continue." turn ahead of it,
	// leaving the real system instruction sitting in the message body after
	// "Continue." instead of actually reaching Gemini's systemInstruction —
	// extract it into the real system field first.
	if first, ok := msgsList[0].(map[string]interface{}); ok && first["role"] == "system" {
		// system can also be an array of blocks carrying its own
		// cache_control — overwriting it outright with a plain string
		// would silently discard that, so the array case appends a block
		// instead of replacing. An audit caught that the ORIGINAL code
		// skipped this whole extraction (including removing messages[0]
		// from msgsList) whenever system was already an array — leaving
		// a stray role:"system" message sitting in the conversation,
		// which nothing downstream of this function knows how to
		// translate (Gemini only has user/model), while the real system
		// array stayed untouched. Now handled either way: append to an
		// array, concatenate onto a string, or set fresh — but always
		// remove messages[0] once its content (if any) has somewhere to
		// go.
		sysText := strings.TrimSpace(extractPlainText(first["content"]))
		if sysText != "" {
			if existingArr, ok := claudeReqJSON["system"].([]interface{}); ok {
				claudeReqJSON["system"] = append(existingArr, map[string]interface{}{"type": "text", "text": sysText})
			} else if existing, _ := claudeReqJSON["system"].(string); existing != "" {
				claudeReqJSON["system"] = existing + "\n\n" + sysText
			} else {
				claudeReqJSON["system"] = sysText
			}
		}
		msgsList = msgsList[1:]
		modified = true
	}
	if len(msgsList) == 0 {
		claudeReqJSON["messages"] = []interface{}{}
		return modified, truncatedAny
	}

	var newMsgs []map[string]interface{}

	// Step 1: Eager Truncation and Text Block Normalization
	for _, mItem := range msgsList {
		m, ok := mItem.(map[string]interface{})
		if !ok {
			continue
		}
		role, _ := m["role"].(string)
		if role == "" {
			role = "user"
		}

		// SanitizeMessagesAndGraph runs more than once per request (translator.go
		// calls it before AND after TruncateConversationToFit, and again after
		// loop-guard nudges) — its own first pass normalizes content below from
		// raw-JSON []interface{} to []map[string]interface{}. Re-wrap that back
		// into []interface{} so the type switch below (which only recognizes
		// string/[]interface{}) still matches it on a second/third pass, instead
		// of falling to default and replacing the whole message with "...".
		if normed, ok := m["content"].([]map[string]interface{}); ok {
			reraw := make([]interface{}, len(normed))
			for i, item := range normed {
				reraw[i] = item
			}
			m["content"] = reraw
		}

		var normBlocks []map[string]interface{}

		switch content := m["content"].(type) {
		case string:
			if strings.Contains(content, "\x00") {
				content = strings.ReplaceAll(content, "\x00", "")
				modified = true
			}
			cleaned := StripPollutedToolCallText(content)
			if cleaned != content {
				content = cleaned
				modified = true
			}
			if strings.TrimSpace(content) == "" || strings.TrimSpace(content) == "." {
				content = "..."
				modified = true
			}
			normBlocks = append(normBlocks, map[string]interface{}{
				"type": "text",
				"text": content,
			})
		case []interface{}:
			for _, bItem := range content {
				b, ok := bItem.(map[string]interface{})
				if !ok {
					continue
				}
				bType, _ := b["type"].(string)

				if bType == "text" {
					txt, _ := b["text"].(string)
					if strings.Contains(txt, "\x00") {
						txt = strings.ReplaceAll(txt, "\x00", "")
						b["text"] = txt
						modified = true
					}
					cleaned := StripPollutedToolCallText(txt)
					if cleaned != txt {
						txt = cleaned
						b["text"] = txt
						modified = true
					}
					if strings.TrimSpace(txt) == "" || (role == "assistant" && strings.TrimSpace(txt) == ".") {
						modified = true
						continue // drop empty or hollow dot text block
					}
					normBlocks = append(normBlocks, b)
				} else if bType == "tool_result" {
					// Eager truncate tool_result
					switch c := b["content"].(type) {
					case string:
						if strings.Contains(c, "\x00") {
							c = strings.ReplaceAll(c, "\x00", "")
							b["content"] = c
							modified = true
						}
						if truncated, did := TruncateMiddle(c); did {
							b["content"] = truncated
							modified = true
							truncatedAny = true
						}
					case []interface{}:
						for _, pItem := range c {
							p, ok := pItem.(map[string]interface{})
							if ok && p["type"] == "text" {
								if pt, ok := p["text"].(string); ok {
									if strings.Contains(pt, "\x00") {
										pt = strings.ReplaceAll(pt, "\x00", "")
										p["text"] = pt
										modified = true
									}
									if truncated, did := TruncateMiddle(pt); did {
										p["text"] = truncated
										modified = true
										// An audit caught this sibling case
										// (array-of-blocks tool_result)
										// missing the truncatedAny=true its
										// string-content sibling above
										// already sets — the dashboard's
										// "Shrunk" flag silently never fired
										// for this content shape.
										truncatedAny = true
									}
								}
							}
						}
					}
					normBlocks = append(normBlocks, b)
				} else {
					normBlocks = append(normBlocks, b)
				}
			}
			if len(normBlocks) == 0 {
				normBlocks = append(normBlocks, map[string]interface{}{
					"type": "text",
					"text": "...",
				})
				modified = true
			}
		default:
			normBlocks = append(normBlocks, map[string]interface{}{
				"type": "text",
				"text": "...",
			})
			modified = true
		}

		newMsgs = append(newMsgs, map[string]interface{}{
			"role":    role,
			"content": normBlocks,
		})
	}

	if len(newMsgs) == 0 {
		return false, false
	}

	// Step 2: Ensure First Message is User
	if newMsgs[0]["role"] != "user" {
		userTurn := map[string]interface{}{
			"role": "user",
			"content": []map[string]interface{}{
				{"type": "text", "text": "Continue."},
			},
		}
		newMsgs = append([]map[string]interface{}{userTurn}, newMsgs...)
		modified = true
	}

	// Step 3: Merge Consecutive Same Roles
	var merged []map[string]interface{}
	for _, m := range newMsgs {
		if len(merged) > 0 && merged[len(merged)-1]["role"] == m["role"] {
			targetBlocks := merged[len(merged)-1]["content"].([]map[string]interface{})
			srcBlocks := m["content"].([]map[string]interface{})
			merged[len(merged)-1]["content"] = append(targetBlocks, srcBlocks...)
			modified = true
		} else {
			merged = append(merged, m)
		}
	}
	newMsgs = merged

	// Step 4: Tool Use / Result Graph Consistency
	toolUsesSeen := make(map[string]bool)
	toolResultsSeen := make(map[string]bool)

	for _, m := range newMsgs {
		blocks := m["content"].([]map[string]interface{})
		for _, b := range blocks {
			bType, _ := b["type"].(string)
			if bType == "tool_use" {
				if id, ok := b["id"].(string); ok && id != "" {
					toolUsesSeen[id] = true
				}
			} else if bType == "tool_result" {
				if tid, ok := b["tool_use_id"].(string); ok && tid != "" {
					toolResultsSeen[tid] = true
				}
			}
		}
	}

	// Turn orphaned tool_results into text
	for _, m := range newMsgs {
		blocks := m["content"].([]map[string]interface{})
		for _, b := range blocks {
			if b["type"] == "tool_result" {
				tid, _ := b["tool_use_id"].(string)
				if !toolUsesSeen[tid] {
					b["type"] = "text"
					cVal := b["content"]
					txt := toolResultPlainText(cVal)
					if txt != "" {
						b["text"] = fmt.Sprintf("[Tool output %s]: %s", tid, txt)
					} else {
						b["text"] = fmt.Sprintf("[Tool output %s]", tid)
					}
					delete(b, "tool_use_id")
					delete(b, "content")
					modified = true
				}
			}
		}
	}

	// Synthesize answers for abandoned tool_uses, immediately after the
	// assistant turn that made them — not dumped at the end of the whole
	// conversation. Gemini requires each functionCall to be closed by its
	// functionResponse in the very next turn; stacking every recovered stub
	// at the tail broke that for any orphaned call that wasn't in the very
	// last assistant turn.
	var rebuilt []map[string]interface{}
	skipNext := false
	for i, m := range newMsgs {
		if skipNext {
			skipNext = false
			continue
		}
		rebuilt = append(rebuilt, m)

		if m["role"] != "assistant" {
			continue
		}
		mContent := m["content"].([]map[string]interface{})
		var toolUseBlocks []map[string]interface{}
		var synths []map[string]interface{}
		nextIsUser := i+1 < len(newMsgs) && newMsgs[i+1]["role"] == "user"

		nextTurnResultIDs := make(map[string]bool)
		if nextIsUser {
			for _, b := range newMsgs[i+1]["content"].([]map[string]interface{}) {
				if b["type"] == "tool_result" {
					if tid, ok := b["tool_use_id"].(string); ok && tid != "" {
						nextTurnResultIDs[tid] = true
					}
				}
			}
		}

		for _, b := range mContent {
			if b["type"] != "tool_use" {
				continue
			}
			toolUseBlocks = append(toolUseBlocks, b)
			// Gemini requires every functionCall in turn N to be answered in turn N+1.
			// If not answered in newMsgs[i+1], synthesize a recovery stub immediately
			// so turn N+1 matches, even if the real result arrives delayed in turn N+3.
			if id, ok := b["id"].(string); ok && id != "" && !nextTurnResultIDs[id] {
				synths = append(synths, map[string]interface{}{
					"type":        "tool_result",
					"tool_use_id": id,
					"content":     "(recovered by proxy)",
				})
			}
		}
		if len(toolUseBlocks) == 0 {
			// This assistant turn made no tool_use at all, so it has
			// nothing to answer — but the client may still have put a
			// tool_result in the following user turn answering some OLDER
			// call (e.g. a delayed subagent handback, or truncation/an
			// intervening text-only turn separating it from its real
			// functionCall). toolUsesSeen above only checked that the id
			// exists SOMEWHERE in history, not in THIS immediately
			// preceding turn, so it survives that pass unmodified. Gemini
			// requires every functionResponse to match a functionCall in
			// the immediately preceding turn, so it still has to be
			// scrubbed to text below, same as the partial-match case.
			hasStrayResult := false
			if nextIsUser {
				for _, b := range newMsgs[i+1]["content"].([]map[string]interface{}) {
					if b["type"] == "tool_result" {
						hasStrayResult = true
						break
					}
				}
			}
			if !hasStrayResult {
				continue
			}
		}
		// Even with nothing to synthesize, the client is allowed to send
		// genuine tool_result blocks in any order (parallel tool calls can
		// finish out of order) — Gemini still requires functionResponse
		// parts to mirror the original functionCall order, so this needs
		// to run whenever m made any tool_use call, not just when recovery
		// stubs are needed.
		if len(synths) == 0 && !nextIsUser {
			continue
		}
		modified = true
		if nextIsUser {
			nextBlocks := newMsgs[i+1]["content"].([]map[string]interface{})
			// Gemini requires functionResponse parts in the SAME order as
			// the functionCall parts they answer — always prepending the
			// synthetic recovery stubs ahead of the real tool_results (as
			// this used to do unconditionally) only happened to be correct
			// when the abandoned call was the FIRST one m made. If it was a
			// middle or later call, the synthetic stub still landed first,
			// putting the responses out of order relative to m's original
			// tool_use sequence. Rebuild in that original order instead.
			realResults := make(map[string]map[string]interface{}, len(nextBlocks))
			var otherBlocks []map[string]interface{}
			for _, b := range nextBlocks {
				if b["type"] == "tool_result" {
					if tid, ok := b["tool_use_id"].(string); ok && tid != "" {
						realResults[tid] = b
						continue
					}
				}
				otherBlocks = append(otherBlocks, b)
			}
			synthByID := make(map[string]map[string]interface{}, len(synths))
			for _, s := range synths {
				if tid, ok := s["tool_use_id"].(string); ok {
					synthByID[tid] = s
				}
			}
			ordered := make([]map[string]interface{}, 0, len(nextBlocks)+len(synths))
			consumed := make(map[string]bool, len(realResults))
			for _, b := range m["content"].([]map[string]interface{}) {
				if b["type"] != "tool_use" {
					continue
				}
				id, _ := b["id"].(string)
				if id == "" {
					continue
				}
				if real, ok := realResults[id]; ok {
					ordered = append(ordered, real)
					consumed[id] = true
				} else if synth, ok := synthByID[id]; ok {
					ordered = append(ordered, synth)
				}
			}
			// A tool_result whose tool_use_id doesn't match any tool_use in
			// m (answering a call from further back than the immediately
			// preceding assistant turn) cannot be emitted as a live functionResponse,
			// because Gemini strictly enforces that every functionResponse in turn N
			// must match a functionCall in turn N-1. Convert it into a plain text block
			// so the information is preserved without triggering a Gemini protobuf 400 error.
			for tid, real := range realResults {
				if !consumed[tid] {
					cVal := real["content"]
					txt := toolResultPlainText(cVal)
					textDesc := fmt.Sprintf("[Delayed tool output for %s]: %s", tid, txt)
					if txt == "" {
						textDesc = fmt.Sprintf("[Delayed tool output for %s]", tid)
					}
					ordered = append(ordered, map[string]interface{}{
						"type": "text",
						"text": textDesc,
					})
				}
			}
			ordered = append(ordered, otherBlocks...)
			newMsgs[i+1]["content"] = ordered
			rebuilt = append(rebuilt, newMsgs[i+1])
			skipNext = true
		} else {
			rebuilt = append(rebuilt, map[string]interface{}{
				"role":    "user",
				"content": synths,
			})
		}
	}
	newMsgs = rebuilt

	// Collapse repeated identical Read results: re-reading the exact same
	// unchanged file is the single most common shape of the "unproductive
	// read-loop" LoopGuard exists to catch, and every repeat read bloats
	// the very context that pushes the conversation toward auto-compaction
	// — which then erases the detailed memory of having read the file,
	// so the model reads it again. Confirmed real-world pattern
	// (independent investigation into Claude Code's own read-loop issue).
	// Folding duplicates into a short marker (instead of letting them sit
	// in full every single turn) also reduces the temptation to re-read,
	// since the model no longer has to scroll past a wall of old file text
	// to find anything new.
	if dedupedRepeatedReads(newMsgs) {
		modified = true
	}

	// Gemini's generateContent requires the conversation to end on a "user"
	// turn; Anthropic allows the last message to be "assistant" (prefill:
	// the client is asking the model to continue its own prior text, e.g. a
	// literal "{" to force JSON output). The tool_use rebuild above already
	// appends a trailing user turn whenever the last assistant message had
	// tool calls — this catches the plain-text prefill case that leaves
	// otherwise. Gemini has no real prefill mechanism (no way to hand a
	// model a prefix and have it literally continue those exact tokens),
	// so a bare "Continue." gave it no idea a prefix even existed — it
	// would restate a "Certainly, here's the continuation..." preamble and
	// repeat the prefix instead of picking up from it, breaking anything
	// parsing the output as a single JSON value. Telling it the exact
	// prefix text and to continue from it without repeating it is the best
	// Gemini-side approximation available.
	if len(newMsgs) > 0 && newMsgs[len(newMsgs)-1]["role"] == "assistant" {
		prefillText := extractPlainText(newMsgs[len(newMsgs)-1]["content"])
		nudgeText := "Continue."
		if prefillText != "" {
			nudgeText = fmt.Sprintf(
				"[SYSTEM NOTE: Your previous turn ended mid-output with this exact text, which the user already has: %q — continue DIRECTLY from the next character. Do not repeat, restate, or add any preamble before it; your reply must pick up exactly where that text left off.]",
				prefillText,
			)
		}
		newMsgs = append(newMsgs, map[string]interface{}{
			"role": "user",
			"content": []map[string]interface{}{
				{"type": "text", "text": nudgeText},
			},
		})
		modified = true
	}

	if modified {
		var genericList []interface{}
		for _, m := range newMsgs {
			// Every step above (normalization, merging, synthesis) works
			// against the concrete []map[string]interface{} content type
			// internally — e.g. Step 3's merge-consecutive-roles type
			// assertion right above requires exactly that, with no ", ok"
			// to fall back on. Converting any earlier would panic there.
			// Only now, on the way OUT to claudeReqJSON (which every other
			// consumer — context_window.go's conservativeTokenEstimate,
			// translator.go's lastTurnHadFailedEdit, loopguard.go's
			// InjectUserNudge — type-switches expecting the same
			// []interface{} encoding/json itself would have produced) do
			// we convert. Skipping this step is what silently blinded all
			// three of those on every request past the first
			// SanitizeMessagesAndGraph call once it started running more
			// than once per request.
			if blocks, ok := m["content"].([]map[string]interface{}); ok {
				m["content"] = blocksToInterfaceSlice(blocks)
			}
			genericList = append(genericList, m)
		}
		claudeReqJSON["messages"] = genericList
	}

	return modified, truncatedAny
}
