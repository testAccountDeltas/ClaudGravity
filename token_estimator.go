package main

import (
	"encoding/base64"
	"encoding/json"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
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
	var width, height int
	if err == nil && cfg.Width > 0 && cfg.Height > 0 {
		width, height = cfg.Width, cfg.Height
	} else {
		// A Round 15 audit caught that Go's stdlib image package has no
		// built-in WebP decoder (no golang.org/x/image/webp import, by
		// design — not worth a new dependency for this), so
		// image.DecodeConfig always fails on a WebP image and every one
		// fell back to the flat 1600-token floor regardless of its real
		// size — a 32x32 icon and a 4K WebP screenshot billed identically.
		// Parse just the 30-byte RIFF/WebP header by hand instead.
		if w, h, ok := decodeWebPDimensions(cleaned); ok {
			width, height = w, h
		} else {
			return anthropicImageTokenFloor
		}
	}
	w, h := float64(width), float64(height)
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

// decodeWebPDimensions hand-parses just enough of a WebP file's RIFF
// container to read its pixel dimensions, covering all three real payload
// shapes (simple lossy VP8, lossless VP8L, and the extended VP8X container
// used for alpha/animation) per Google's documented WebP container format.
// Decodes only the first 30 bytes of base64 — never the full image.
func decodeWebPDimensions(b64Data string) (width, height int, ok bool) {
	buf := make([]byte, 30)
	n, _ := io.ReadFull(base64.NewDecoder(base64.StdEncoding, strings.NewReader(b64Data)), buf)
	if n < 30 {
		return 0, 0, false
	}
	if string(buf[0:4]) != "RIFF" || string(buf[8:12]) != "WEBP" {
		return 0, 0, false
	}
	fourCC := string(buf[12:16])
	payload := buf[20:]
	switch fourCC {
	case "VP8 ": // simple lossy: 3-byte frame tag, 3-byte start code, then 14-bit W/H
		if len(payload) < 10 || payload[3] != 0x9d || payload[4] != 0x01 || payload[5] != 0x2a {
			return 0, 0, false
		}
		w := int(payload[6]) | int(payload[7]&0x3f)<<8
		h := int(payload[8]) | int(payload[9]&0x3f)<<8
		return w, h, w > 0 && h > 0
	case "VP8L": // lossless: 1-byte signature 0x2f, then a packed 32-bit LE field
		if len(payload) < 5 || payload[0] != 0x2f {
			return 0, 0, false
		}
		val := uint32(payload[1]) | uint32(payload[2])<<8 | uint32(payload[3])<<16 | uint32(payload[4])<<24
		w := int(val&0x3FFF) + 1
		h := int((val>>14)&0x3FFF) + 1
		return w, h, true
	case "VP8X": // extended container: 1 flags byte, 3 reserved, then 24-bit W-1/H-1
		if len(payload) < 10 {
			return 0, 0, false
		}
		w := int(payload[4]) | int(payload[5])<<8 | int(payload[6])<<16
		h := int(payload[7]) | int(payload[8])<<8 | int(payload[9])<<16
		return w + 1, h + 1, true
	}
	return 0, 0, false
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
// A Round 15 audit caught that the unanchored /Count\s+(\d+) pattern below
// matches the FIRST /Count it finds anywhere in the raw bytes — a PDF with
// a document outline (table of contents / bookmarks) declares its own
// "/Type /Outlines /Count N" dict, which commonly appears before the real
// page tree's "/Type /Pages /Count N" in the byte stream, and N there is
// the outline's own entry count (often 1), not the page count. A 100-page
// PDF with one top-level bookmark entry was estimated at 258 tokens
// instead of 25,800 — a 100x undercount that let TruncateConversationToFit
// treat the request as tiny and pass it straight to Google, which then
// rejected it for genuinely exceeding the context window. Anchoring to
// "/Type /Pages" specifically (allowing either key order within a
// reasonable window) avoids matching the outline's unrelated /Count.
var pdfPageCountPattern = regexp.MustCompile(`/Type\s*/Pages[\s\S]{0,250}?/Count\s+(\d+)`)

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
	matches := pdfPageCountPattern.FindAllSubmatch(raw, -1)
	if matches == nil {
		return 0, false
	}
	// A nested/intermediate /Pages node (a PDF can have a tree of them, not
	// just one flat root) carries its own smaller subtree /Count — take the
	// largest match found, which is always the root's total page count.
	maxPages := 0
	for _, m := range matches {
		pages, err := strconv.Atoi(string(m[1]))
		if err != nil || pages <= 0 {
			continue
		}
		if pages > maxPages {
			maxPages = pages
		}
	}
	if maxPages == 0 {
		return 0, false
	}
	return maxPages * geminiPDFTokensPerPage, true
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
									srcType := item.Get("source.type").String()
									switch {
									case srcType == "url":
										// A Round 15 audit caught this fell straight to
										// estimateImageTokens's flat 1600-token floor (source.data
										// is empty for a url source) — translator.go never
										// fetches the URL, it substitutes a short
										// "[Attached by URL: <url>]" text placeholder instead, so
										// the honest cost scales with that string's length.
										if u := item.Get("source.url").String(); u != "" {
											total += estimateApproxTokens("[Attached by URL: " + u + "]")
										} else {
											total += estimateApproxTokens("[Attached by URL: ]")
										}
									case strings.ToLower(item.Get("source.media_type").String()) == "image/svg+xml":
										// A Round 15 audit caught this never checked media_type
										// the way the "document" branch below already does for
										// the identical shape — an SVG sent as an "image" block
										// is XML TEXT (Gemini's inlineData can't take SVG, so
										// translator.go sends it as text instead), not a raster
										// image; image.DecodeConfig can't parse XML and this fell
										// to the flat 1600-token floor regardless of real size, so
										// a 200KB SVG (~50,000 real tokens of markup) was priced
										// 30x too low.
										dataStr := item.Get("source.data").String()
										total += len(dataStr) * 3 / 4 / 2
									default:
										total += estimateImageTokens(item.Get("source.data").String())
									}
								case "document":
									src := item.Get("source")
									if src.Get("type").String() == "base64" {
										mediaType := strings.ToLower(src.Get("media_type").String())
										dataStr := src.Get("data").String()
										var est int
										pageCountTrusted := false
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
											if pageEst, ok := estimatePDFTokensFromPageCount(dataStr); ok {
												pageCountTrusted = true
												if pageEst > est {
													est = pageEst
												}
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
										//
										// A follow-up audit caught the floor was ALSO applied
										// when estimatePDFTokensFromPageCount succeeded — a
										// genuine 1-page PDF (real cost ~258 tokens) got
										// floored up to 2500 even though the page count was
										// directly read from the PDF's own structure, not
										// guessed. Skip the floor specifically when that
										// trustworthy signal was found.
										if !pageCountTrusted && (mediaType == "application/pdf" || mediaType == "") && est < minPDFTokenFloor {
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
									// A Round 15 audit caught that a bare
									// string array element (content as
									// ["line 1\n", "line 2\n"], not typed
									// blocks) has no "type"/"text" key at
									// all — item.Get("text") on a plain
									// string scalar returns empty, scoring
									// it 0 tokens and silently undercounting
									// a 100KB tool output entirely.
									if item.Type == gjson.String {
										total += estimateApproxTokens(item.String())
									} else {
										total += estimateApproxTokens(item.Get("text").String())
									}
								}
								return true
							})
						} else if rc.Exists() {
							// A Round 15 audit caught that a tool_result's
							// content given as a bare JSON OBJECT (neither
							// a string nor an array — e.g. a tool
							// returning structured {"text": "...", ...}
							// directly) matched neither branch above and
							// fell straight through to scoring 0 tokens.
							// Price its raw JSON the same conservative way
							// a single untyped content block is priced.
							total += estimateApproxTokens(rc.Raw)
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
							pageCountTrusted := false
							if mediaType == "application/pdf" || mediaType == "" {
								est = len(dataStr) / 500
								if pageEst, ok := estimatePDFTokensFromPageCount(dataStr); ok {
									pageCountTrusted = true
									if pageEst > est {
										est = pageEst
									}
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
							// document case above (and context_window.go) —
							// skipped when the page-count signal was trusted,
							// same reasoning as that site.
							if !pageCountTrusted && (mediaType == "application/pdf" || mediaType == "") && est < minPDFTokenFloor {
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
						// A Round 15 audit caught this priced the full raw
						// encrypted "data" blob (can run tens of thousands
						// of characters) instead of the fixed 26-byte
						// "[Prior reasoning redacted]" placeholder
						// translator.go always substitutes before this
						// ever reaches Gemini — on every turn, not just the
						// most recent one (unlike real "thinking", which
						// the isLastAssistant gate correctly protects for
						// the thought-signature validator's sake).
						total += estimateApproxTokens("[Prior reasoning redacted]")
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
			rawSchema := t.Get("input_schema").Raw
			// A Round 15 audit caught this priced input_schema as-is —
			// Anthropic's server-side built-in tools (bash_20250124,
			// text_editor_20250728, computer_20250124, ...) are declared
			// by the client with NO input_schema at all (translator.go's
			// builtinAnthropicToolSchema already knows this and injects
			// the real reference schema before ever building the Gemini
			// payload), so every one of these scored 0 extra tokens here
			// even though Anthropic's own server fills in and bills for a
			// real ~150-token parameter schema underneath.
			if strings.TrimSpace(rawSchema) == "" || rawSchema == "null" || rawSchema == "{}" {
				if builtin, ok := builtinAnthropicToolSchema(t.Get("type").String()); ok {
					rawSchema = builtin
				}
			}
			total += estimateApproxTokens(rawSchema)
			return true
		})
	}

	if total < 1 {
		return 1
	}
	return total
}
