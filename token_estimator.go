package main

import (
	"encoding/json"
	"math"
	"strings"

	"github.com/tidwall/gjson"
)

// estimateContentSourceText extracts the real text from a document block
// whose source.type == "content" (Anthropic's other text-document form,
// alongside "base64"/"text"/"url" — "content" is either a literal string or
// an array of {"type":"text","text":...} blocks). An audit caught that
// callers previously fell back to reading src.Get("data").String()
// regardless of source type — "content" documents have no "data" field at
// all, so that silently priced them at 0 tokens, letting a genuinely large
// document slip through token accounting unnoticed and overflow Gemini's
// real context window with no warning.
func estimateContentSourceText(src gjson.Result) string {
	contentField := src.Get("content")
	if contentField.Type == gjson.String {
		return contentField.String()
	}
	if contentField.IsArray() {
		var sb strings.Builder
		contentField.ForEach(func(_, block gjson.Result) bool {
			if block.Get("type").String() == "text" {
				sb.WriteString(block.Get("text").String())
			} else if block.Type == gjson.String {
				sb.WriteString(block.String())
			}
			return true
		})
		return sb.String()
	}
	return ""
}

// detectRussianConversation checks the latest user-role message for Cyrillic
// text, to pick which language gateway-synthesized strings use (the
// grounding-sources footer, currently) — this text bypasses the model
// entirely (built by the gateway itself after the model has already
// finished, or instead of calling it), so the model has no chance to match
// its own output language to it. The user's own latest message is the
// freshest signal for "what language is this conversation in right now."
func detectRussianConversation(bodyBytes []byte) bool {
	msgs := gjson.GetBytes(bodyBytes, "messages")
	if !msgs.IsArray() {
		return false
	}
	arr := msgs.Array()
	for i := len(arr) - 1; i >= 0; i-- {
		m := arr[i]
		if m.Get("role").String() != "user" {
			continue
		}
		var text string
		content := m.Get("content")
		if content.Type == gjson.String {
			text = content.String()
		} else if content.IsArray() {
			content.ForEach(func(_, block gjson.Result) bool {
				if block.Get("type").String() == "text" {
					text = block.Get("text").String()
					return false
				}
				return true
			})
		}
		if text == "" {
			continue
		}
		cyrillic := 0
		for _, r := range text {
			if r >= 0x0400 && r <= 0x04FF {
				cyrillic++
				if cyrillic >= 3 {
					return true
				}
			}
		}
		return false
	}
	return false
}

// estimateApproxTokens computes a weighted character-to-token approximation
// using Anthropic's BPE weight characteristics (ASCII, digits, punctuation, non-ASCII/CJK/Cyrillic).
func estimateApproxTokens(text string) int {
	if text == "" {
		return 0
	}

	runes := []rune(text)
	length := len(runes)
	if length == 0 {
		return 0
	}
	if length < 5 {
		return int(math.Ceil(float64(length) / 3.0))
	}

	var regularAscii, digits, symbols, nonASCII int
	for _, r := range runes {
		switch {
		case r >= 0x80:
			nonASCII++
		case r >= '0' && r <= '9':
			digits++
		case (r >= '!' && r <= '/') || (r >= ':' && r <= '@') || (r >= '[' && r <= '`') || (r >= '{' && r <= '~'):
			symbols++
		default:
			regularAscii++
		}
	}

	estimated := int(math.Ceil(
		float64(regularAscii)/4.5 +
			float64(digits)/2.0 +
			float64(symbols)/1.5 +
			float64(nonASCII)/1.5,
	))

	if estimated < 1 {
		return 1
	}
	return estimated
}

func estimateJSONTokens(v interface{}) int {
	if v == nil {
		return 0
	}
	b, err := json.Marshal(v)
	if err != nil {
		return 0
	}
	return estimateApproxTokens(string(b))
}

// estimateApproxClaudeTokens parses an entire Anthropic Messages API payload
// (system instructions, messages, images, tool definitions) and computes accurate token count.
func estimateApproxClaudeTokens(bodyBytes []byte) int {
	root := gjson.ParseBytes(bodyBytes)
	total := 0

	// 1. System Prompt
	if sys := root.Get("system"); sys.Exists() {
		if sys.Type == gjson.String {
			total += estimateApproxTokens(sys.String())
		} else if sys.IsArray() {
			sys.ForEach(func(_, block gjson.Result) bool {
				total += estimateApproxTokens(block.Get("text").String())
				return true
			})
		}
	}

	// 2. Messages History
	if msgs := root.Get("messages"); msgs.IsArray() {
		msgs.ForEach(func(_, m gjson.Result) bool {
			// Per-message framing overhead (~3 tokens)
			total += 3

			c := m.Get("content")
			if c.Type == gjson.String {
				total += estimateApproxTokens(c.String())
			} else if c.IsArray() {
				c.ForEach(func(_, part gjson.Result) bool {
					pType := part.Get("type").String()
					switch pType {
					case "text":
						total += estimateApproxTokens(part.Get("text").String())
					case "thinking":
						total += estimateApproxTokens(part.Get("thinking").String())
					case "tool_use":
						total += estimateApproxTokens(part.Get("name").String())
						total += estimateApproxTokens(part.Get("input").Raw)
					case "tool_result":
						rc := part.Get("content")
						if rc.Type == gjson.String {
							total += estimateApproxTokens(rc.String())
						} else if rc.IsArray() {
							rc.ForEach(func(_, item gjson.Result) bool {
								switch item.Get("type").String() {
								case "image":
									total += 1600
								case "document":
									src := item.Get("source")
									if src.Get("type").String() == "base64" {
										mediaType := strings.ToLower(src.Get("media_type").String())
										dataStr := src.Get("data").String()
										var est int
										if mediaType == "application/pdf" || mediaType == "" {
											// PDF token density (~258 tokens/page, ~1 token per
											// 500 base64 chars) — len/2 here would price a 2MB
											// scanned PDF at ~1.33M tokens, see context_window.go.
											est = len(dataStr) / 500
										} else {
											// A base64 text document (code, logs, CSV) has none
											// of a PDF's image-heavy byte density — decode to raw
											// text and price at the normal char/token ratio, or
											// this undercounts a 1MB text file by ~100x.
											decodedBytes := len(dataStr) * 3 / 4
											est = decodedBytes / 2
										}
										// The 1600-token floor approximates a minimum
										// "page" cost for PDFs — applying it to a small
										// text document (a short code snippet, a tiny
										// CSV) overstates its real size for no reason.
										if (mediaType == "application/pdf" || mediaType == "") && est < 1600 {
											est = 1600
										}
										total += est
									} else if src.Get("type").String() == "content" {
										total += estimateApproxTokens(estimateContentSourceText(src))
									} else {
										total += estimateApproxTokens(src.Get("data").String())
									}
								default:
									total += estimateApproxTokens(item.Get("text").String())
								}
								return true
							})
						}
					case "image":
						// Rough image token cost (typical Anthropic resolution ~1600 tokens)
						total += 1600
					case "document":
						src := part.Get("source")
						if src.Get("type").String() == "base64" {
							mediaType := strings.ToLower(src.Get("media_type").String())
							dataStr := src.Get("data").String()
							var est int
							if mediaType == "application/pdf" || mediaType == "" {
								est = len(dataStr) / 500
							} else {
								decodedBytes := len(dataStr) * 3 / 4
								est = decodedBytes / 2
							}
							// Same PDF-only floor as the tool_result document
							// case above (and context_window.go) — applying
							// it to every media type inflated a small base64
							// text/code/CSV document's displayed estimate for
							// no reason. This is the top-level "document"
							// content block's own copy of that same bug; the
							// nested tool_result one was already fixed.
							if (mediaType == "application/pdf" || mediaType == "") && est < 1600 {
								est = 1600
							}
							total += est
						} else if src.Get("type").String() == "content" {
							total += estimateApproxTokens(estimateContentSourceText(src))
						} else {
							total += estimateApproxTokens(src.Get("data").String())
						}
					case "redacted_thinking":
						total += estimateApproxTokens(part.Get("data").String())
					}
					return true
				})
			}
			return true
		})
	}

	// 3. Tool Definitions
	if tools := root.Get("tools"); tools.IsArray() {
		tools.ForEach(func(_, t gjson.Result) bool {
			total += estimateApproxTokens(t.Get("name").String())
			total += estimateApproxTokens(t.Get("description").String())
			total += estimateApproxTokens(t.Get("input_schema").Raw)
			return true
		})
	}

	if total < 1 {
		return 1
	}
	return total
}
