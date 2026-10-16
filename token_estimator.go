package main

import (
	"encoding/json"
	"math"

	"github.com/tidwall/gjson"
)

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
								total += estimateApproxTokens(item.Get("text").String())
								return true
							})
						}
					case "image":
						// Rough image token cost (typical Anthropic resolution ~1600 tokens)
						total += 1600
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
