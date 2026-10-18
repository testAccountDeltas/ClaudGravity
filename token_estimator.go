package main

import (
	"encoding/base64"
	"encoding/json"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"regexp"
	"strconv"
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
	// An audit caught that a client sending a full data URI
	// ("data:image/png;base64,...") instead of bare base64 made
	// DecodeString fail outright on the non-base64 "data:image/png;base64,"
	// prefix, silently falling to the flat floor regardless of the
	// image's real size — translator.go already strips this same prefix
	// before decoding for the actual translation path.
	// An audit caught that decoding the ENTIRE image into a []byte just to
	// read its header dimensions allocated the full binary payload in
	// heap memory for no reason — image.DecodeConfig only ever reads as
	// many bytes as the format's header actually needs (a PNG's IHDR
	// chunk, a JPEG's SOF marker) and stops there on its own; feeding it
	// a streaming base64 decoder avoids materializing the rest of a
	// large image (a multi-MB screenshot) at all.
	cleaned := stripBase64Whitespace(stripDataURIPrefix(b64Data))
	cfg, _, err := image.DecodeConfig(base64.NewDecoder(base64.StdEncoding, strings.NewReader(cleaned)))
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

// pdfPageCountPattern looks for an uncompressed PDF page-tree's /Count key
// directly in the raw decoded bytes — this is a best-effort heuristic, not
// a real PDF parser: a PDF using compressed object streams (common for
// larger/newer-generator PDFs) hides this inside a FlateDecode stream this
// won't see, in which case the scan simply finds nothing and callers fall
// back to their existing estimate. It never makes an estimate WORSE: it
// only replaces a generic density-based guess with Gemini's real flat
// 258-tokens/page billing on PDFs where the page count happens to be
// directly readable in the raw bytes — verified against a real exim: see
// git history for the test that validated the regex against a
// constructed, syntactically valid uncompressed PDF.
var pdfPageCountPattern = regexp.MustCompile(`/Count\s+(\d+)`)

// geminiPDFTokensPerPage is Gemini's real documented flat per-page PDF
// billing, independent of how text-dense or image-dense that page is —
// the len/500 density-based estimate below this function's callers can
// undercount a text-sparse/vector-graphics-heavy PDF by several times
// since it has far less base64 "weight" per page than a scanned/rasterized
// one despite costing Gemini the same flat per-page rate either way.
const geminiPDFTokensPerPage = 258

// estimatePDFTokensFromPageCount decodes base64 PDF data and looks for a
// plaintext /Count N. Returns (0, false) if decoding fails or no page
// count is found in the raw bytes, letting the caller fall back to its
// own density-based estimate.
func estimatePDFTokensFromPageCount(b64Data string) (int, bool) {
	// An audit caught that a client sending a full data: URI (e.g.
	// "data:application/pdf;base64,...") instead of raw base64 — some
	// browser-based clients do this even though Anthropic's own spec
	// expects raw base64 — made base64.StdEncoding.DecodeString fail
	// outright (":" "/" ";" "," aren't valid base64 characters), silently
	// losing the chance at a real page count and falling back to the
	// less accurate density estimate instead. stripDataURIPrefix already
	// exists for this exact reason elsewhere in the package.
	raw, err := base64.StdEncoding.DecodeString(stripBase64Whitespace(stripDataURIPrefix(b64Data)))
	if err != nil || len(raw) == 0 {
		return 0, false
	}
	m := pdfPageCountPattern.FindSubmatch(raw)
	if m == nil {
		return 0, false
	}
	pages, err := strconv.Atoi(string(m[1]))
	if err != nil || pages <= 0 {
		return 0, false
	}
	return pages * geminiPDFTokensPerPage, true
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
		// An audit caught that commit 40b25b9's fix (context_window.go's
		// conservativeTokenEstimate pricing old thinking blocks at 0,
		// since translator.go guarantees they're stripped before ever
		// reaching Gemini) was never mirrored here — this SEPARATE
		// estimator feeds message_start's reported input_tokens and
		// count_tokens, and kept summing every historical thinking
		// block's real size, over-reporting by 50-100k tokens on a long
		// agentic session for phantom tokens that were never sent.
		msgArr := msgs.Array()
		lastAssistantIdx := -1
		for i := len(msgArr) - 1; i >= 0; i-- {
			if msgArr[i].Get("role").String() == "assistant" {
				lastAssistantIdx = i
				break
			}
		}
		msgs.ForEach(func(idxResult, m gjson.Result) bool {
			isLastAssistant := m.Get("role").String() == "assistant" && int(idxResult.Int()) == lastAssistantIdx
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
						if isLastAssistant {
							total += estimateApproxTokens(part.Get("thinking").String())
						}
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
											// An audit caught this density guess undercounts a
											// text-sparse/vector PDF (far fewer base64 bytes per
											// page than a scanned one, same real per-page Gemini
											// cost) by several times — prefer a real page count
											// when it's readable directly in the raw bytes.
											if pageEst, ok := estimatePDFTokensFromPageCount(dataStr); ok && pageEst > est {
												est = pageEst
											}
										} else if strings.HasPrefix(mediaType, "image/") && mediaType != "image/svg+xml" {
											// An audit caught a "document" block whose
											// media_type is an image mimetype fell into the
											// text-density branch below, pricing binary base64
											// bytes as if decoded text — a 2MB image inflated to
											// ~750,000 "tokens" instead of the real ~1600.
											est = estimateImageTokens(dataStr)
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
								if pageEst, ok := estimatePDFTokensFromPageCount(dataStr); ok && pageEst > est {
									est = pageEst
								}
							} else if strings.HasPrefix(mediaType, "image/") && mediaType != "image/svg+xml" {
								// Same image-mispriced-as-text fix as the
								// tool_result document case above.
								est = estimateImageTokens(dataStr)
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
						if isLastAssistant {
							total += estimateApproxTokens(part.Get("data").String())
						}
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
