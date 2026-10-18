package main

import (
	"encoding/base64"
	"encoding/json"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"strings"

	"github.com/tidwall/gjson"
)

// anthropicImageTokenFloor is the flat fallback used when the actual pixel
// dimensions can't be determined (corrupt/truncated/unsupported-format
// base64, or a non-base64 source like a URL this gateway never fetches) —
// close to the real cost of a typical mid-size screenshot, so an estimate
// that has to fall back isn't wildly off either direction.
const anthropicImageTokenFloor = 1600

// anthropicImageMaxSide is the side-length Anthropic's own documented
// pipeline downscales an image to before pricing it — a larger image is
// billed as if resized to fit this, not at its original resolution.
const anthropicImageMaxSide = 1568

// estimateImageTokens computes Anthropic's own documented image-pricing
// formula, ceil(width*height/750) after capping the longer side to 1568px,
// instead of a flat per-image guess that treats a tiny icon and a full
// desktop screenshot identically. Decodes only the image header (no full
// pixel decode) via image.DecodeConfig. Falls back to the flat floor when
// dimensions can't be read.
func estimateImageTokens(b64Data string) int {
	if b64Data == "" {
		return anthropicImageTokenFloor
	}
	raw, err := base64.StdEncoding.DecodeString(stripBase64Whitespace(b64Data))
	if err != nil || len(raw) == 0 {
		return anthropicImageTokenFloor
	}
	cfg, _, err := image.DecodeConfig(strings.NewReader(string(raw)))
	if err != nil || cfg.Width <= 0 || cfg.Height <= 0 {
		return anthropicImageTokenFloor
	}
	w, h := float64(cfg.Width), float64(cfg.Height)
	if maxSide := math.Max(w, h); maxSide > anthropicImageMaxSide {
		scale := anthropicImageMaxSide / maxSide
		w *= scale
		h *= scale
	}
	tokens := int(math.Ceil(w * h / 750))
	if tokens < 1 {
		tokens = 1
	}
	return tokens
}

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
									total += estimateImageTokens(item.Get("source.data").String())
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
										// minPDFTokenFloor (context_window.go) approximates
										// a minimum "page" cost for PDFs — applying it to a
										// small text document (a short code snippet, a tiny
										// CSV) overstates its real size for no reason. An
										// audit caught this file and context_window.go had
										// drifted to two DIFFERENT floor values (1600 here,
										// 2500 there) despite being the same approximation;
										// now shares one constant.
										if (mediaType == "application/pdf" || mediaType == "") && est < minPDFTokenFloor {
											est = minPDFTokenFloor
										}
										total += est
									} else if src.Get("type").String() == "content" {
										total += estimateApproxTokens(estimateContentSourceText(src))
									} else if src.Get("type").String() == "url" {
										// An audit caught this priced a
										// url-sourced document at 0 tokens
										// (src.data is empty for a url
										// source) — but the real content
										// sent to Gemini is a short
										// "[Attached by URL: <url>]" text
										// placeholder (translator.go never
										// fetches the URL), so the honest
										// estimate scales with the URL
										// string's length, matching
										// context_window.go's equivalent fix.
										total += estimateApproxTokens("[Attached by URL: " + src.Get("url").String() + "]")
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
						// An audit cross-referenced Anthropic's own documented
						// image-pricing formula (ceil(w*h/750) after capping
						// the longer side to 1568px) against this flat 1600
						// guess — real images vary from a small icon (a few
						// hundred tokens) to a full desktop screenshot (3000+),
						// so decode the actual dimensions instead of guessing.
						total += estimateImageTokens(part.Get("source.data").String())
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
							// Same shared minPDFTokenFloor as the tool_result
							// document case above (and context_window.go).
							if (mediaType == "application/pdf" || mediaType == "") && est < minPDFTokenFloor {
								est = minPDFTokenFloor
							}
							total += est
						} else if src.Get("type").String() == "content" {
							total += estimateApproxTokens(estimateContentSourceText(src))
						} else if src.Get("type").String() == "url" {
							// Same url-source fix as the tool_result
							// document case above.
							total += estimateApproxTokens("[Attached by URL: " + src.Get("url").String() + "]")
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
