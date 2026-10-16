package main

import (
	"fmt"
	"regexp"
	"strings"
	"unicode/utf8"
)

const ToolOutputCharLimit = 65536 // 64 KB limit

var (
	pollutedToolCallTextPattern = regexp.MustCompile(`(?i)\[Called tool [^\]]*\]`)
	multiNewlinePattern         = regexp.MustCompile(`\n{3,}`)
)

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
	half := ToolOutputCharLimit / 2

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

		var normBlocks []map[string]interface{}

		switch content := m["content"].(type) {
		case string:
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
									if truncated, did := TruncateMiddle(pt); did {
										p["text"] = truncated
										modified = true
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
					b["text"] = fmt.Sprintf("[Tool output %s]: %v", tid, cVal)
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
		var synths []map[string]interface{}
		for _, b := range m["content"].([]map[string]interface{}) {
			if b["type"] == "tool_use" {
				if id, ok := b["id"].(string); ok && id != "" && !toolResultsSeen[id] {
					synths = append(synths, map[string]interface{}{
						"type":        "tool_result",
						"tool_use_id": id,
						"content":     "(recovered by proxy)",
					})
				}
			}
		}
		if len(synths) == 0 {
			continue
		}
		modified = true
		if i+1 < len(newMsgs) && newMsgs[i+1]["role"] == "user" {
			nextBlocks := newMsgs[i+1]["content"].([]map[string]interface{})
			merged := make([]map[string]interface{}, 0, len(synths)+len(nextBlocks))
			merged = append(merged, synths...)
			merged = append(merged, nextBlocks...)
			newMsgs[i+1]["content"] = merged
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

	// Gemini's generateContent requires the conversation to end on a "user"
	// turn; Anthropic allows the last message to be "assistant" (prefill:
	// the client is asking the model to continue its own prior text). The
	// tool_use rebuild above already appends a trailing user turn whenever
	// the last assistant message had tool calls — this catches the plain-
	// text prefill case that leaves otherwise.
	if len(newMsgs) > 0 && newMsgs[len(newMsgs)-1]["role"] == "assistant" {
		newMsgs = append(newMsgs, map[string]interface{}{
			"role": "user",
			"content": []map[string]interface{}{
				{"type": "text", "text": "Continue."},
			},
		})
		modified = true
	}

	if modified {
		var genericList []interface{}
		for _, m := range newMsgs {
			genericList = append(genericList, m)
		}
		claudeReqJSON["messages"] = genericList
	}

	return modified, truncatedAny
}


