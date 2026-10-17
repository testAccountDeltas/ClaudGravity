package main

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// currentLocalTimeMinuteSecondRe zeroes the minutes/seconds of an ISO 8601
// timestamp (the "current local time: 2026-10-17T11:56:07+03:00" line
// Claude Code puts in its "# Environment" block), quantizing it to the hour.
// Keeping the line verbatim kept the model honestly date/time-aware but
// changed the system prompt's bytes every single second, invalidating
// Google's byte-prefix cache on every turn; quantizing to the hour keeps the
// model's sense of "now" accurate enough for any real use (git log date
// filtering, "is it business hours") while the system prompt — and the
// cache built on it — stays byte-stable for 3600 seconds at a time instead
// of 1.
var currentLocalTimeMinuteSecondRe = regexp.MustCompile(`(T\d{2}):\d{2}:\d{2}`)

// claudeCodeSelfIDRe matches only the self-identification clause itself —
// the real line continues straight on with real behavioral instructions
// ("Given the user's message, you should use the tools available to
// complete the task. Complete the task fully...") that a whole-line
// replacement in ScrubBillingHeader used to discard along with it.
var claudeCodeSelfIDRe = regexp.MustCompile(`(?i)you are (an agent for )?claude code, anthropic's official cli( for claude)?\.?`)

// DropSearchTools is stripped from the declared tools when
// tenant.StripSearchTools is true (default: true — see README.md's
// "Почему strip_search_tools по умолчанию true" for the full reasoning).
// Deliberate, not a forgotten default: Grep/Glob's schema (pattern, path,
// glob, output_mode, -i/-n/-A/-B/-C, context, multiline, head_limit,
// offset — only "pattern" required) gives Gemini real room to drop the
// required field or confuse which field a flag belongs in, and unlike
// Edit (RepairEditToolCall) there is no defensive repair pass for these
// two yet to catch and fix a malformed call before it reaches the client.
// Without them, Gemini falls back to Bash for search — a predictable,
// visible failure mode (quoting/escaping issues on Windows) instead of a
// silently malformed structured call. The native Antigravity client
// (agy.exe) isn't counter-evidence either way: its own tool palette has
// no Grep/Glob-shaped tool at all — it searches via its own run_command
// (shell), not through this specific schema.
//
// This isn't just theory: a sibling project (opencode, same Gemini 3.x
// models) independently confirmed and live-tested the exact root cause —
// Gemini reliably mishandles specialized multi-field search tools
// (misplaces or drops the required "pattern" field) while handling one
// general shell tool (a plain command string) correctly. Verified 3/3 and
// on a real ~39-call project run with zero errors once grep/glob were
// denied. Same fix there: deny the specialized tools, shell-only.
var DropSearchTools = map[string]bool{
	"Grep": true,
	"Glob": true,
}

// lastTurnHadFailedEdit reports whether the most recent assistant/user turn
// pair in this request carries an Edit (or str_replace_editor/update/
// notebookedit) tool_use answered by a tool_result with is_error:true —
// the earliest point this gateway can see that outcome at all, since the
// client only reports it on its NEXT request.
func lastTurnHadFailedEdit(reqMap map[string]interface{}) bool {
	msgsRaw, ok := reqMap["messages"].([]interface{})
	if !ok || len(msgsRaw) < 2 {
		return false
	}
	last, ok := msgsRaw[len(msgsRaw)-1].(map[string]interface{})
	if !ok || last["role"] != "user" {
		return false
	}
	prev, ok := msgsRaw[len(msgsRaw)-2].(map[string]interface{})
	if !ok || prev["role"] != "assistant" {
		return false
	}

	editToolIDs := make(map[string]bool)
	if blocks, ok := prev["content"].([]interface{}); ok {
		for _, bi := range blocks {
			b, ok := bi.(map[string]interface{})
			if !ok || b["type"] != "tool_use" {
				continue
			}
			name, _ := b["name"].(string)
			switch strings.ToLower(name) {
			case "edit", "str_replace_editor", "update", "notebookedit":
				if id, ok := b["id"].(string); ok && id != "" {
					editToolIDs[id] = true
				}
			}
		}
	}
	if len(editToolIDs) == 0 {
		return false
	}

	if blocks, ok := last["content"].([]interface{}); ok {
		for _, bi := range blocks {
			b, ok := bi.(map[string]interface{})
			if !ok || b["type"] != "tool_result" {
				continue
			}
			tid, _ := b["tool_use_id"].(string)
			isErr, _ := b["is_error"].(bool)
			if isErr && editToolIDs[tid] {
				return true
			}
		}
	}
	return false
}

// isAllHexDigits backs the toolu_<name>_<nanots>_<seq> id-recovery parse in
// ConvertClaudeRequestToGemini below — nextToolCallID (sse.go) hex-encodes
// its timestamp/sequence segments to stay within Anthropic's 64-char
// tool_use.id limit, so the trailing-numeric-segment check here has to
// accept hex digits too, not just decimal ones.
func isAllHexDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, c := range s {
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

// sealFunctionResponseTurn enforces Gemini's "a turn carrying a
// functionResponse must contain ONLY functionResponse parts" rule (the same
// constraint already handled for LoopGuard nudges in loopguard.go): if parts
// mixes functionResponse entries with anything else — plain text, images —
// the text is folded into the first functionResponse's response.user_note.
// Images are NOT nested under functionResponse.parts — multiple independent
// reports agree Google's real FunctionResponse message has exactly 3 fields
// (name, response, id) and no "parts", so that previously silently lost any
// screenshot/plot a tool returned. Returns the now-pure functionResponse
// entries, plus any image parts the caller must emit as a separate
// following turn (role "user", pure inlineData — unambiguously valid
// Gemini syntax, unlike the removed functionResponse.parts guess).
func sealFunctionResponseTurn(parts []string) (sealed []string, followUpImages []string) {
	var fnResp, other []string
	for _, p := range parts {
		if gjson.Get(p, "functionResponse").Exists() {
			fnResp = append(fnResp, p)
		} else {
			other = append(other, p)
		}
	}
	if len(fnResp) == 0 || len(other) == 0 {
		return parts, nil
	}

	var note strings.Builder
	for _, p := range other {
		if t := gjson.Get(p, "text"); t.Exists() {
			if note.Len() > 0 {
				note.WriteString("\n")
			}
			note.WriteString(t.String())
		} else if gjson.Get(p, "inlineData").Exists() {
			followUpImages = append(followUpImages, p)
		}
	}

	target := fnResp[0]
	if note.Len() > 0 {
		target, _ = sjson.Set(target, "functionResponse.response.user_note", note.String())
	}
	fnResp[0] = target
	return fnResp, followUpImages
}

// ConvertClaudeRequestToGemini converts an Anthropic Messages API payload into
// a Google Gemini streamGenerateContent payload with full sanitization and tiering.
func ConvertClaudeRequestToGemini(claudeJSON []byte, tenant TenantConfig, reqClass string, agentKey string, sessionID string, instanceID string) ([]byte, string, int, bool, map[string]bool, error) {
	// loopKey disambiguates both across sessions and across parallel
	// subagents within one session: agentKey alone (e.g. "sub:general-
	// purpose") is shared by every concurrent instance of that subagent
	// type, in every session the gateway is serving, and even bare "main"
	// was one single key shared by every session's main thread. sessionID
	// scopes it to one conversation; instanceID (stable per subagent run,
	// since it's the agent-id header, not the per-batch prompt-id — see
	// ParseAgentInfo's comment) further separates concurrent subagents of
	// the same type within that same session.
	loopKey := agentKey + "|" + sessionID + "|" + instanceID
	// Parse as generic map for in-place AST transformations
	var reqMap map[string]interface{}
	if err := json.Unmarshal(claudeJSON, &reqMap); err != nil {
		return nil, "", 0, false, nil, fmt.Errorf("invalid json: %w", err)
	}

	// 1. Dynamic Tiering & Model Effort Resolution
	modelName, budget := ResolveTieringAndEffort(reqMap, reqClass)
	if modelName == "" {
		modelName = tenant.DefaultModel
	}

	// 1.5 Context-window-aware truncation: drop oldest history if it would
	// exceed the REAL context window of the Gemini sub-model tiering just
	// picked above (see context_window.go for why this can't be handled by
	// the client's static CLAUDE_CODE_MAX_CONTEXT_TOKENS alone).
	//
	// requestedMaxOutputHint must mirror the SAME inflation step 7 below
	// applies to requestedMax (thinking budget forces max_tokens up to
	// budget+4096) — otherwise this reserves too little room for the
	// response and under-truncates, exactly like step 7's own clamp against
	// modelMeta.MaxOutput.
	var requestedMaxOutputHint int64 = 8192
	if mt, ok := reqMap["max_tokens"].(float64); ok && mt > 0 {
		requestedMaxOutputHint = int64(mt)
	}
	if budget > 0 && requestedMaxOutputHint <= int64(budget) {
		requestedMaxOutputHint = int64(budget) + 4096
	} else if budget < 0 {
		// Must mirror step 7's own ModeAuto inflation below (maxOutputTokens
		// raised to the model's full ceiling, since thinking and the visible
		// answer share that one pool under ModeAuto) — otherwise this
		// reserves only ~8192 tokens of headroom while the real request
		// reserves up to 64000, and a conversation sitting near the model's
		// context ceiling gets under-truncated here only to blow past
		// Google's real token wall once the inflated output budget is added.
		if earlyMeta, ok := globalModelCatalog.GetModel(modelName); ok && earlyMeta.MaxOutput > 0 {
			requestedMaxOutputHint = earlyMeta.MaxOutput
		} else {
			requestedMaxOutputHint = 64000
		}
	}
	if dropped, truncated := TruncateConversationToFit(reqMap, modelName, requestedMaxOutputHint); truncated {
		LogGateway("[ContextWindow] dropped %d oldest message(s) to fit %s's context window [%s]", dropped, modelName, agentKey)
	}

	// 2. Loop-Guard Nudges (Read-Loop & Oscillation)
	hasModifyingTools := false
	isPlanMode := false
	if toolsArr, ok := reqMap["tools"].([]interface{}); ok {
		for _, t := range toolsArr {
			if tMap, isMap := t.(map[string]interface{}); isMap {
				name := strings.ToLower(fmt.Sprint(tMap["name"]))
				switch name {
				case "edit", "write", "notebookedit", "str_replace_editor":
					hasModifyingTools = true
				case "exitplanmode":
					// Claude Code declares Edit/Write alongside ExitPlanMode
					// even while Plan Mode is active — the tool catalog
					// doesn't shrink, the system prompt just instructs the
					// model not to call them. Their mere presence must not
					// count as "modifying tools available" here, or every
					// Plan Mode investigation (which is Read/Grep-heavy by
					// design) gets the tight main-agent read-loop threshold
					// instead of the relaxed explore one.
					isPlanMode = true
				}
			}
		}
	}
	if isPlanMode {
		hasModifyingTools = false
	}
	_, readThreshold := readLoopLimits(agentKey, hasModifyingTools)
	if hasModifyingTools {
		if level := globalLoopGuard.CheckAndResetReadStreak(loopKey, readThreshold); level > 0 {
			nudgeText := NudgeReadLoop
			if level >= 2 {
				nudgeText = NudgeReadLoopEscalated
			}
			InjectUserNudge(reqMap, nudgeText)
			LogGateway(".. read-loop window >= %d Read calls [%s] level=%d -> injecting stop-nudge", readThreshold, loopKey, level)
		}
	}
	if globalLoopGuard.CheckAndResetOscillation(loopKey) {
		InjectUserNudge(reqMap, NudgeOscillation)
		LogGateway(".. tool-call oscillation [%s] -> injecting loop-nudge", loopKey)
	}
	if editFailStreak := globalLoopGuard.RecordEditResult(loopKey, lastTurnHadFailedEdit(reqMap)); editFailStreak >= 3 {
		InjectUserNudge(reqMap, NudgeEditFailureLoop)
		LogGateway(".. %d consecutive Edit failures [%s] -> injecting edit-failure-loop nudge", editFailStreak, loopKey)
	}

	// 3. Full AST Sanitization (Middle Truncation 64KB, Non-empty sentinels, role repair, tool pairing)
	_, wasShrunk := SanitizeMessagesAndGraph(reqMap)

	// Re-serialize sanitized request
	sanitizedBytes, err := json.Marshal(reqMap)
	if err != nil {
		return nil, "", 0, false, nil, fmt.Errorf("re-marshal sanitized json: %w", err)
	}

	root := gjson.ParseBytes(sanitizedBytes)
	out := "{}"

	// 4. System Instruction (Scrubbing billing headers)
	systemPrompt := ""
	if sys := root.Get("system"); sys.Exists() {
		if sys.IsArray() {
			var sb strings.Builder
			sys.ForEach(func(_, block gjson.Result) bool {
				txt := block.Get("text").String()
				if cleaned := ScrubBillingHeader(txt); cleaned != "" {
					sb.WriteString(cleaned)
					sb.WriteString("\n")
				}
				return true
			})
			systemPrompt = strings.TrimSpace(sb.String())
		} else if sys.Type == gjson.String {
			systemPrompt = ScrubBillingHeader(sys.String())
		}
	}
	if systemPrompt != "" {
		// Per Google's REST spec, system_instruction is just {"parts": [...]}
		// — role is only meaningful on conversation turns in "contents",
		// not here. CloudCode's internal protobuf parser tolerates the
		// extra field (confirmed: this ran with it present for this whole
		// session without issue), but the public Gemini Developer API path
		// isn't worth the same risk for a field that was never required.
		out, _ = sjson.Set(out, "system_instruction.parts.0.text", systemPrompt)
	}

	// 5. Messages Translation
	toolIDToName := make(map[string]string)
	// Validates any recovered-by-heuristic fnName below against what's
	// actually declared — the id-parsing fallback assumes our own
	// "toolu_<name>_<nanots>_<seq>" format (see nextToolCallID), but a
	// native Anthropic id like "toolu_01XfD..." has no underscore at all,
	// so isAllDigits never trims anything and the WHOLE opaque id string
	// becomes "fnName" — non-empty, so the old empty-string fallback never
	// caught it, and Google rejected a functionResponse.name that matches
	// no declared tool.
	declaredToolNames := make(map[string]bool)
	if toolsArr := root.Get("tools"); toolsArr.IsArray() {
		toolsArr.ForEach(func(_, t gjson.Result) bool {
			if n := t.Get("name").String(); n != "" {
				declaredToolNames[n] = true
			}
			return true
		})
	}
	messages := root.Get("messages")

	if messages.IsArray() {
		contentIdx := 0
		var lastRole string

		messages.ForEach(func(_, msg gjson.Result) bool {
			role := msg.Get("role").String()
			geminiRole := "user"
			if role == "assistant" {
				geminiRole = "model"
			}

			// Pre-pass: map tool_use IDs to function names
			if parts := msg.Get("content"); parts.IsArray() {
				parts.ForEach(func(_, p gjson.Result) bool {
					if p.Get("type").String() == "tool_use" {
						id := p.Get("id").String()
						name := p.Get("name").String()
						if id != "" && name != "" {
							toolIDToName[id] = name
						}
					}
					return true
				})
			}

			var partsJSON []string

			if content := msg.Get("content"); content.IsArray() {
				content.ForEach(func(_, p gjson.Result) bool {
					ptype := p.Get("type").String()
					switch ptype {
					case "text":
						txt := p.Get("text").String()
						if strings.TrimSpace(txt) != "" {
							part := `{"text":""}`
							part, _ = sjson.Set(part, "text", txt)
							partsJSON = append(partsJSON, part)
						}
					case "image", "document":
						// "document" (PDF, added to Anthropic's API in Sep
						// 2024) was falling into no case at all and being
						// silently dropped — Gemini takes it the exact same
						// way as an image: inlineData with the real
						// mimeType (application/pdf here instead of
						// image/*).
						src := p.Get("source")
						switch src.Get("type").String() {
						case "base64":
							mediaType := src.Get("media_type").String()
							b64Data := src.Get("data").String()
							if b64Data != "" {
								part := `{"inlineData":{"mimeType":"","data":""}}`
								part, _ = sjson.Set(part, "inlineData.mimeType", mediaType)
								part, _ = sjson.Set(part, "inlineData.data", b64Data)
								partsJSON = append(partsJSON, part)
							}
						case "text":
							// A plain-text document source (source.data is
							// the literal text, not base64) was silently
							// dropped entirely — only "base64" was ever
							// handled. Gemini has no document type at all;
							// the real content is just text either way.
							if txt := src.Get("data").String(); txt != "" {
								part := `{"text":""}`
								part, _ = sjson.Set(part, "text", txt)
								partsJSON = append(partsJSON, part)
							}
						}
					case "tool_use":
						name := p.Get("name").String()
						callID := p.Get("id").String()
						argsRaw := p.Get("input").Raw
						if argsRaw == "" {
							argsRaw = "{}"
						}
						// Prefer the real signature this exact call was
						// originally emitted with (cached in sse.go/
						// server.go when we first sent it) over the
						// skip-validator bypass — Anthropic's tool_use
						// schema has no field to round-trip it through the
						// client itself, unlike "thinking" blocks.
						sig := getToolSignature(callID)
						if sig == "" {
							sig = "skip_thought_signature_validator"
						}
						part := `{"thoughtSignature":"","functionCall":{"name":"","args":{}}}`
						part, _ = sjson.Set(part, "thoughtSignature", sig)
						part, _ = sjson.Set(part, "functionCall.name", name)
						if callID != "" {
							part, _ = sjson.Set(part, "functionCall.id", callID)
						}
						part, _ = sjson.SetRaw(part, "functionCall.args", argsRaw)
						partsJSON = append(partsJSON, part)

					case "tool_result":
						toolUseID := p.Get("tool_use_id").String()
						fnName := toolIDToName[toolUseID]
						if fnName == "" {
							// Map miss (its tool_use fell outside a
							// truncated context window, or a subagent
							// history desync) previously fell back to the
							// literal string "tool" — Gemini validates
							// functionResponse.name against the declared
							// tools and rejects anything not in that list
							// with a guaranteed 400. Recover the real name
							// from our own id format first: "toolu_<name>_
							// <nanots>_<seq>" (see nextToolCallID in
							// sse.go). Can't just split on "_" since a tool
							// name may itself contain underscores — strip
							// the trailing two purely-numeric segments
							// instead of guessing which piece is the name.
							if rest, ok := strings.CutPrefix(toolUseID, "toolu_"); ok {
								segs := strings.Split(rest, "_")
								end := len(segs)
								// nextToolCallID always appends EXACTLY two trailing
								// segments (a hex nanosecond timestamp, a hex sequence
								// number) after the name -- an open-ended strip loop
								// could also eat a real trailing segment of the tool's
								// own name that happens to look hex ("tool_v2_1": "1" is
								// valid hex, stripping it left "tool_v2" instead of the
								// real name). Stop at exactly two.
								for stripped := 0; end > 0 && stripped < 2 && isAllHexDigits(segs[end-1]); stripped++ {
									end--
								}
								if end > 0 {
									fnName = strings.Join(segs[:end], "_")
								}
							}
							// Validate the recovered name against what's
							// actually declared, regardless of where it
							// came from — a native Anthropic id has no
							// underscore at all, so the parse above never
							// trims anything and fnName ends up as the
							// whole opaque id string (non-empty, so it
							// would otherwise sail straight past here).
							if fnName != "" && !declaredToolNames[fnName] {
								fnName = ""
							}
							if fnName == "" {
								// Last resort: any declared tool at least
								// matches Google's validation; "tool" never
								// does.
								fnName = root.Get("tools.0.name").String()
							}
							if fnName == "" {
								fnName = "tool"
							}
						}

						resText := ""
						var imageParts []string
						c := p.Get("content")
						if c.IsArray() {
							var sb strings.Builder
							c.ForEach(func(_, item gjson.Result) bool {
								switch item.Get("type").String() {
								case "text":
									sb.WriteString(item.Get("text").String())
								case "image", "document":
									// Previously dropped entirely (only "text"
									// blocks were read), silently losing any
									// screenshot/plot/PDF a tool returned.
									src := item.Get("source")
									switch src.Get("type").String() {
									case "base64":
										mediaType := src.Get("media_type").String()
										data := src.Get("data").String()
										if mediaType != "" && data != "" {
											imgPart := `{"inlineData":{"mimeType":"","data":""}}`
											imgPart, _ = sjson.Set(imgPart, "inlineData.mimeType", mediaType)
											imgPart, _ = sjson.Set(imgPart, "inlineData.data", data)
											imageParts = append(imageParts, imgPart)
										}
									case "text":
										if txt := src.Get("data").String(); txt != "" {
											sb.WriteString(txt)
										}
									}
								}
								return true
							})
							resText = sb.String()
						} else if c.Exists() {
							resText = c.String()
						}

						// is_error was previously ignored, so a failed tool
						// call looked identical to a successful one to the
						// model — it had no signal to recognize the failure
						// and retry or change approach.
						responseField := "result"
						if p.Get("is_error").Bool() {
							responseField = "error"
						}
						part := `{"functionResponse":{"name":"","response":{}}}`
						part, _ = sjson.Set(part, "functionResponse.name", fnName)
						if toolUseID != "" {
							part, _ = sjson.Set(part, "functionResponse.id", toolUseID)
						}
						part, _ = sjson.Set(part, "functionResponse.response."+responseField, resText)
						partsJSON = append(partsJSON, part)
						// Append any images from THIS tool_result as sibling
						// top-level parts (not nested under
						// functionResponse.parts, which real Gemini
						// FunctionResponse messages don't have — Google
						// silently drops unknown fields, losing the image
						// entirely). sealFunctionResponseTurn below sees
						// these as "other" parts mixed into the turn and
						// moves them into a separate following plain "user"
						// turn, the same way it already handles a stray
						// image/document Claude content block sitting
						// alongside a tool_result at the message level.
						partsJSON = append(partsJSON, imageParts...)

					case "thinking":
						thoughtTxt := p.Get("thinking").String()
						if thoughtTxt != "" {
							if role != "assistant" {
								// Anthropic's spec never puts a "thinking"
								// block in a user message, but a client
								// built against a looser convention (not
								// Claude Code itself, but one of the other
								// Anthropic-API-compatible web/Electron
								// clients this gateway also serves) could
								// still send one. Google rejects "thought"
								// outright on a non-model turn ("thought is
								// only supported for model turns") — render
								// it as plain text instead of risking a 400
								// on the whole request over one stray block.
								part := `{"text":""}`
								part, _ = sjson.Set(part, "text", thoughtTxt)
								partsJSON = append(partsJSON, part)
								return true
							}
							// Prefer the real signature the client echoed back
							// (now that sse.go actually sends one via
							// signature_delta) over the skip-validator bypass;
							// Google warns that permanently skipping the
							// validator can degrade reasoning quality on long
							// thinking chains.
							sig := p.Get("signature").String()
							if sig == "" {
								sig = "skip_thought_signature_validator"
							}
							part := `{"thought":true,"thoughtSignature":"","text":""}`
							part, _ = sjson.Set(part, "thoughtSignature", sig)
							part, _ = sjson.Set(part, "text", thoughtTxt)
							partsJSON = append(partsJSON, part)
						}
					case "redacted_thinking":
						// Opaque, encrypted — Gemini has no equivalent and
						// there's nothing real to translate (this only
						// shows up at all if history was imported from a
						// genuine Anthropic/Claude conversation, since this
						// gateway's own responses never produce one). Keep
						// the turn's structure intact with a placeholder
						// instead of silently dropping it, which could
						// otherwise leave an assistant turn with zero parts.
						if role != "assistant" {
							part := `{"text":"[Prior reasoning redacted]"}`
							partsJSON = append(partsJSON, part)
							return true
						}
						part := `{"thought":true,"thoughtSignature":"skip_thought_signature_validator","text":"[Prior reasoning redacted]"}`
						partsJSON = append(partsJSON, part)
					}
					return true
				})
			} else if content.Type == gjson.String {
				txt := content.String()
				if strings.TrimSpace(txt) != "" {
					part := `{"text":""}`
					part, _ = sjson.Set(part, "text", txt)
					partsJSON = append(partsJSON, part)
				}
			}

			if len(partsJSON) == 0 {
				partsJSON = append(partsJSON, `{"text":"..."}`)
			}
			var followUpImages []string
			partsJSON, followUpImages = sealFunctionResponseTurn(partsJSON)

			if geminiRole == lastRole && contentIdx > 0 {
				targetIdx := contentIdx - 1
				// This message's own parts are already pure (sealed above),
				// but merging them into the PREVIOUS turn can still produce
				// a mix if that turn is pure functionResponse and this one
				// is pure text (or vice versa) — both the same Gemini role,
				// e.g. a tool_result-only turn immediately followed by a
				// separate plain-text user turn. Re-seal the combined set.
				existingRaw := gjson.Get(out, fmt.Sprintf("contents.%d.parts", targetIdx))
				var combined []string
				existingRaw.ForEach(func(_, v gjson.Result) bool {
					combined = append(combined, v.Raw)
					return true
				})
				combined = append(combined, partsJSON...)
				var mergedFollowUp []string
				combined, mergedFollowUp = sealFunctionResponseTurn(combined)
				followUpImages = append(followUpImages, mergedFollowUp...)
				if geminiRole == "model" {
					// Google Antigravity strictly enforces that all thought parts must
					// precede any other parts in a model turn, even after merging consecutive model turns.
					var thoughtParts, otherParts []string
					for _, pjson := range combined {
						if gjson.Get(pjson, "thought").Bool() {
							thoughtParts = append(thoughtParts, pjson)
						} else {
							otherParts = append(otherParts, pjson)
						}
					}
					combined = append(thoughtParts, otherParts...)
				}
				combinedJSON := "[" + strings.Join(combined, ",") + "]"
				out, _ = sjson.SetRaw(out, fmt.Sprintf("contents.%d.parts", targetIdx), combinedJSON)
			} else {
				if geminiRole == "model" {
					// Google Antigravity strictly enforces that all thought parts must
					// precede any other parts in a model turn.
					var thoughtParts, otherParts []string
					for _, pjson := range partsJSON {
						if gjson.Get(pjson, "thought").Bool() {
							thoughtParts = append(thoughtParts, pjson)
						} else {
							otherParts = append(otherParts, pjson)
						}
					}
					partsJSON = append(thoughtParts, otherParts...)
				}
				out, _ = sjson.Set(out, fmt.Sprintf("contents.%d.role", contentIdx), geminiRole)
				for _, pjson := range partsJSON {
					out, _ = sjson.SetRaw(out, fmt.Sprintf("contents.%d.parts.-1", contentIdx), pjson)
				}
				lastRole = geminiRole
				contentIdx++
			}

			if len(followUpImages) > 0 {
				// The turn right before this one (the functionResponse
				// turn these images came out of) is ALSO role "user" —
				// Gemini's tools/results always live on the user side,
				// the same way Claude's tool_result does. Appending this
				// image turn as role "user" too put two consecutive
				// "user" entries in contents[], which Gemini's strict
				// user/model alternation rejects outright with 400
				// "multiturn requests alternate between user and model
				// roles" — turning every tool call that returns an image
				// into a guaranteed crash instead of just a lost image.
				// A one-line synthetic model turn in between restores
				// alternation.
				out, _ = sjson.Set(out, fmt.Sprintf("contents.%d.role", contentIdx), "model")
				ackPart := `{"text":"Tool output received. Processing visual data."}`
				out, _ = sjson.SetRaw(out, fmt.Sprintf("contents.%d.parts.-1", contentIdx), ackPart)
				contentIdx++

				out, _ = sjson.Set(out, fmt.Sprintf("contents.%d.role", contentIdx), "user")
				notePart := `{"text":"Visual output from the previous tool call:"}`
				out, _ = sjson.SetRaw(out, fmt.Sprintf("contents.%d.parts.-1", contentIdx), notePart)
				for _, pjson := range followUpImages {
					out, _ = sjson.SetRaw(out, fmt.Sprintf("contents.%d.parts.-1", contentIdx), pjson)
				}
				lastRole = "user"
				contentIdx++
			}

			return true
		})
	}

	// 6. Tools Declaration
	syntheticReasonTools := make(map[string]bool)
	// Tracks which tool names actually made it into Gemini's
	// functionDeclarations (post strip_search_tools/web_search filtering)
	// — NOT the same set as declaredToolNames above, which reflects
	// everything the CLIENT originally declared before any of that
	// filtering. tool_choice's allowedFunctionNames guard below needs the
	// post-filter set: Grep/Glob stay in declaredToolNames even when
	// stripped, so checking against that let tool_choice:{"type":"tool",
	// "name":"Grep"} through, 400ing with "allowed_function_names contains
	// 'Grep' which is not in function_declarations".
	sentToolNames := make(map[string]bool)
	if tools := root.Get("tools"); tools.IsArray() {
		var decls []string
		hasWebSearch := false

		tools.ForEach(func(_, t gjson.Result) bool {
			name := t.Get("name").String()
			// Claude Code's real native tool is snake_case "web_search"
			// (confirmed against websearch.go's own fast-path check, which
			// matches that exact string) — "WebSearch"/"GoogleSearch" never
			// matched it, so inline Gemini googleSearch grounding never
			// actually activated for the one real tool name it exists for.
			normName := strings.ToLower(strings.ReplaceAll(name, "_", ""))
			if normName == "websearch" || normName == "googlesearch" {
				hasWebSearch = true
				// Never declare this as a regular custom function too — it
				// isn't one (this gateway has no handler for a "web_search"
				// functionCall; it exists purely to flip hasWebSearch above
				// so native Gemini googleSearch grounding can be used
				// instead). Leaving it in decls meant decls was virtually
				// NEVER empty (Claude Code always declares web_search), so
				// the "only add googleSearch when there are no other tools"
				// check below never actually fired either way.
				return true
			}
			if tenant.StripSearchTools && DropSearchTools[name] {
				return true
			}
			desc := t.Get("description").String()
			rawSchema := t.Get("input_schema").Raw
			schemaRaw := CleanJSONSchemaForAntigravity(name, rawSchema)
			if strings.TrimSpace(schemaRaw) == "" || schemaRaw == "null" {
				// Google rejects an OBJECT-type schema with empty properties
				// ("parameters.properties: should be non-empty for OBJECT
				// type") — route through the same placeholder mechanism
				// CleanJSONSchemaForAntigravity already uses for schemas
				// that end up empty after cleaning, instead of a raw
				// properties:{} that bypasses it entirely.
				schemaRaw, _ = addEmptySchemaPlaceholder(`{"type":"object","properties":{}}`, nil)
			}
			// Remember which tools got our synthetic "reason" placeholder
			// injected (the only way Gemini will accept an argument-less
			// tool's schema) so the caller can strip that key back out of
			// the model's actual function-call args before handing them to
			// Claude Code — a strict MCP tool with additionalProperties:
			// false rejects a "reason" field it never declared.
			if strings.Contains(schemaRaw, placeholderReasonDescription) {
				syntheticReasonTools[name] = true
			}

			decl := `{"name":"","description":"","parameters":{}}`
			decl, _ = sjson.Set(decl, "name", name)
			decl, _ = sjson.Set(decl, "description", desc)
			decl, _ = sjson.SetRaw(decl, "parameters", schemaRaw)
			decls = append(decls, decl)
			sentToolNames[name] = true
			return true
		})

		if len(decls) > 0 {
			out, _ = sjson.SetRaw(out, "tools.0.functionDeclarations", "["+strings.Join(decls, ",")+"]")

			// Tried generationConfig's documented
			// toolConfig.functionCallingConfig.streamFunctionCallArguments
			// to get incremental tool-arg deltas for large calls (e.g.
			// Write with a big file body) instead of one lump near the end
			// of generation. Confirmed via live testing that the internal
			// CloudCode/Antigravity endpoint we use
			// (v1internal:streamGenerateContent) ignores it — the flag
			// reaches the backend correctly but every functionCall part
			// still arrives complete in a single chunk. Removed; this
			// appears to be an inherent property of this specific
			// undocumented endpoint, not something fixable client-side.
		}
		// A prior attempt combined custom functionDeclarations with the
		// built-in googleSearch tool in the same tools array (citing Google
		// docs that Gemini 3 supports this), and a follow-up tried adding
		// toolConfig.includeServerSideToolInvocations:true to satisfy
		// Google's "Please enable tool_config.include_server_side_tool_
		// invocations to use Built-in tools with Function calling" 400 —
		// that flag did NOT fix it live (confirmed: same 400 recurred after
		// deploying it), and the exact right fix for this undocumented
		// internal endpoint isn't known. Reverted to mutually exclusive:
		// real tool execution (Read/Write/Bash/...) matters far more than
		// native search grounding, and declaring both together is what
		// breaks every subagent spawn outright. Revisit only with a live-
		// confirmed fix, not another guess at the field name/path.
		if hasWebSearch && len(decls) == 0 {
			out, _ = sjson.SetRaw(out, "tools.0.googleSearch", "{}")
		}

		// tool_choice was previously dropped entirely, so a client forcing
		// tool use (type:"any") or a specific tool (type:"tool", name:X)
		// silently fell back to Gemini's own default "let the model decide".
		if len(decls) > 0 {
			if tc := root.Get("tool_choice"); tc.Exists() {
				switch tc.Get("type").String() {
				case "any":
					out, _ = sjson.Set(out, "toolConfig.functionCallingConfig.mode", "ANY")
				case "tool":
					if name := tc.Get("name").String(); name != "" && sentToolNames[name] {
						out, _ = sjson.Set(out, "toolConfig.functionCallingConfig.mode", "ANY")
						out, _ = sjson.Set(out, "toolConfig.functionCallingConfig.allowedFunctionNames.0", name)
					}
				case "none":
					out, _ = sjson.Set(out, "toolConfig.functionCallingConfig.mode", "NONE")
				}
			}
		}
	}

	// 7. Thinking & Output Tokens Configuration with Dynamic Model Limits
	modelMeta, hasMeta := globalModelCatalog.GetModel(modelName)

	// budget==0 used to be unreachable (the effort-based switch in
	// tiering.go never produced it) and this block simply skipped itself
	// for it. It's now a real, deliberate signal too — the standard
	// Anthropic "thinking": {"type": "disabled"} request maps to it — so
	// it must still send an explicit thinkingBudget:0 rather than fall
	// through to Gemini's own default (models that think by default would
	// otherwise keep thinking with no cap at all).
	out, _ = sjson.Set(out, "generationConfig.thinkingConfig.includeThoughts", budget != 0)
	if budget > 0 {
		if hasMeta && modelMeta.MinThinkingBudget > 0 && budget < modelMeta.MinThinkingBudget {
			budget = modelMeta.MinThinkingBudget
		}
		out, _ = sjson.Set(out, "generationConfig.thinkingConfig.thinkingBudget", budget)
	} else if budget < 0 {
		// -1 enables ModeAuto unbracketed
		out, _ = sjson.Set(out, "generationConfig.thinkingConfig.thinkingBudget", -1)
	} else {
		out, _ = sjson.Set(out, "generationConfig.thinkingConfig.thinkingBudget", 0)
	}

	// 6.5 Carry over sampling hyperparameters: previously dropped entirely,
	// so a client asking for temperature:0 (deterministic code generation)
	// silently got Gemini's own default (commonly ~1.0) instead.
	if temp := root.Get("temperature"); temp.Exists() {
		out, _ = sjson.Set(out, "generationConfig.temperature", temp.Float())
	}
	if topP := root.Get("top_p"); topP.Exists() {
		out, _ = sjson.Set(out, "generationConfig.topP", topP.Float())
	}
	if topK := root.Get("top_k"); topK.Exists() {
		out, _ = sjson.Set(out, "generationConfig.topK", topK.Int())
	}
	if stopSeqs := root.Get("stop_sequences"); stopSeqs.IsArray() {
		var seqs []string
		for _, s := range stopSeqs.Array() {
			seqs = append(seqs, s.String())
		}
		// Google hard-caps generationConfig.stopSequences at 5 entries —
		// anything past that fails the whole request with a 400
		// INVALID_ARGUMENT instead of just ignoring the extras.
		if len(seqs) > 5 {
			seqs = seqs[:5]
		}
		if len(seqs) > 0 {
			seqsJSON, _ := json.Marshal(seqs)
			out, _ = sjson.SetRaw(out, "generationConfig.stopSequences", string(seqsJSON))
		}
	}

	// 6.6 Structured Outputs: translate Anthropic output_config.format into Gemini responseSchema
	if fmtNode := root.Get("output_config.format"); fmtNode.Exists() {
		if fmtNode.Get("type").String() == "json_schema" {
			out, _ = sjson.Set(out, "generationConfig.responseMimeType", "application/json")
			if schemaNode := fmtNode.Get("schema"); schemaNode.Exists() {
				cleanedSchema := CleanJSONSchemaForGemini(schemaNode.Raw)
				out, _ = sjson.SetRaw(out, "generationConfig.responseSchema", cleanedSchema)
			}
		}
	}

	requestedMax := root.Get("max_tokens").Int()
	if requestedMax <= 0 {
		if hasMeta && modelMeta.MaxOutput > 0 {
			requestedMax = modelMeta.MaxOutput
		} else {
			requestedMax = 8192
		}
	} else if hasMeta && modelMeta.MaxOutput > 0 && requestedMax > modelMeta.MaxOutput {
		// Clamp requested tokens to model's certified ceiling (prevent Google 400 Bad Request)
		requestedMax = modelMeta.MaxOutput
	}

	// Gemini's maxOutputTokens is a single shared ceiling for thinking AND
	// the visible answer combined — unlike Claude's max_tokens, which only
	// caps the visible output. Under ModeAuto (budget == -1) the model can
	// freely spend an unbounded share of that SAME ceiling on invisible
	// thinking before Claude Code's own (often small, e.g. 8192) max_tokens
	// request is even considered; whatever's left can truncate mid
	// tool-call-argument JSON. Give ModeAuto the model's full certified
	// ceiling regardless of what the client asked for — there's no cost
	// downside on this quota-based backend to requesting a higher ceiling
	// the model doesn't end up using.
	if budget < 0 {
		if hasMeta && modelMeta.MaxOutput > 0 {
			requestedMax = modelMeta.MaxOutput
		} else {
			requestedMax = 64000
		}
	}

	// For models with extended thinking enabled (e.g. Claude on Vertex AI):
	// max_tokens MUST be strictly greater than thinking.budget_tokens.
	// If requestedMax <= budget, expand requestedMax so the upstream request is valid.
	if budget > 0 && requestedMax <= int64(budget) {
		ceiling := int64(64000)
		if hasMeta && modelMeta.MaxOutput > 0 {
			ceiling = modelMeta.MaxOutput
		}
		if int64(budget)+4096 <= ceiling {
			requestedMax = int64(budget) + 4096
		} else {
			// If budget is at ceiling, adjust budget downwards so requestedMax can exceed it
			budget = int(ceiling) - 1024
			out, _ = sjson.Set(out, "generationConfig.thinkingConfig.thinkingBudget", budget)
			requestedMax = ceiling
		}
	}

	out, _ = sjson.Set(out, "generationConfig.maxOutputTokens", requestedMax)

	// 8. Uncensored Safety Settings
	safetyCategories := []string{
		"HARM_CATEGORY_HARASSMENT",
		"HARM_CATEGORY_HATE_SPEECH",
		"HARM_CATEGORY_SEXUALLY_EXPLICIT",
		"HARM_CATEGORY_DANGEROUS_CONTENT",
		"HARM_CATEGORY_CIVIC_INTEGRITY",
	}
	for i, cat := range safetyCategories {
		out, _ = sjson.Set(out, fmt.Sprintf("safetySettings.%d.category", i), cat)
		threshold := "OFF"
		if cat == "HARM_CATEGORY_CIVIC_INTEGRITY" {
			threshold = "BLOCK_NONE"
		}
		out, _ = sjson.Set(out, fmt.Sprintf("safetySettings.%d.threshold", i), threshold)
	}

	return []byte(out), modelName, budget, wasShrunk, syntheticReasonTools, nil
}

// ScrubBillingHeader strips out fingerprint headers and billing marks that trigger Google 429/Filter:
// 1. x-anthropic-billing-header
// 2. Canonical Claude Code self-identification preambles that Google CloudCode actively filters
// 3. Environmental noise (gitStatus, Recent commits, fast mode tags) to reduce prompt tokens
func ScrubBillingHeader(txt string) string {
	var cleanLines []string
	skipSection := false
	skipSectionLines := 0

	for _, line := range strings.Split(txt, "\n") {
		trimmed := strings.TrimSpace(line)
		lower := strings.ToLower(trimmed)

		// Filter billing header marks
		if strings.Contains(line, "x-anthropic-billing-header") {
			continue
		}
		if strings.Contains(line, "Claude Agent SDK") || strings.Contains(line, "built on Anthropic") {
			// Same class of bug as the "you are claude code" self-ID fix
			// below: replacing the WHOLE line risked discarding real
			// trailing instructions that continue past the identifying
			// phrase on the same line. Strip just the phrases themselves
			// instead of guessing the full sentence structure.
			scrubbed := strings.ReplaceAll(line, "Claude Agent SDK", "a coding assistant")
			scrubbed = strings.ReplaceAll(scrubbed, "built on Anthropic", "")
			cleanLines = append(cleanLines, scrubbed)
			continue
		}

		// Filter heavy repetitive environment sections — but the section
		// also carries the model's only signal for which OS/shell it's
		// actually running on (Platform/Shell/working directory). Dropping
		// it wholesale previously left Gemini guessing, and it would guess
		// Unix (export, ls, grep) on a Windows box as often as not.
		if trimmed == "# Environment" {
			skipSection = true
			skipSectionLines = 0
			cleanLines = append(cleanLines, line)
			continue
		}
		if skipSection {
			skipSectionLines++
			// Any heading level ends the section, not just another
			// top-level "# " one — a "## Guidelines"/"### Rules" subsection
			// nested under "# Environment" (or following it before the next
			// real top-level heading) matched neither "# " nor the
			// Environment whitelist below, so it and everything after it
			// kept getting silently dropped as if it were still environment
			// noise, sometimes eating the rest of the system prompt. But
			// "# Environment" can also be the LAST heading in the real
			// prompt entirely — no heading of any level ever follows it —
			// in which case that fix alone still ate everything to the end
			// of the prompt (model description, skills catalog, date). The
			// known Environment fields (platform/shell/os/working
			// directory/git-repo/scratchpad) comfortably fit well under 20
			// lines; past that, whatever follows isn't this section
			// anymore even without a heading to prove it.
			if strings.HasPrefix(trimmed, "#") || skipSectionLines > 20 {
				skipSection = false
			} else {
				if strings.Contains(lower, "platform:") ||
					strings.Contains(lower, "shell:") ||
					strings.Contains(lower, "os version:") ||
					strings.Contains(lower, "working directory:") ||
					// Dropped silently like every other unlisted Environment
					// line — but these two are load-bearing: without "is a
					// git repository" Gemini doesn't know git commands are
					// even on the table, and without "scratchpad directory"
					// it has no designated place for temp files and starts
					// littering the repo root or /tmp instead.
					strings.Contains(lower, "is a git repository:") ||
					strings.Contains(lower, "scratchpad directory:") ||
					// Dropping this line left Gemini with no idea what the
					// real current date/year is, so it would default to
					// whatever's baked into its training data — guessing
					// 2023/2024, misdating git log filters, and generally
					// reasoning about "now" incorrectly.
					strings.Contains(lower, "current local time") {
					cleanLines = append(cleanLines, currentLocalTimeMinuteSecondRe.ReplaceAllString(line, "${1}:00:00"))
				}
				continue
			}
		}

		// Strip fast-mode noise markers only — gitStatus:/Recent commits:
		// used to be stripped too, but that section is only ~80-180 tokens
		// and removing it meant Gemini had no idea which branch/files were
		// already changed, so its very first move was burning a Bash
		// round-trip on "git status"/"git log" to ask for information that
		// was already sitting right there — a turn's worth of latency and
		// tool_use/tool_result tokens to "save" a sliver of context.
		if strings.HasPrefix(trimmed, "<fast_mode_info>") ||
			strings.HasPrefix(trimmed, "</fast_mode_info>") {
			continue
		}

		// Normalize self-identification — replace only the identifying
		// clause, keeping whatever real behavioral instructions follow it
		// on the same line.
		if strings.Contains(lower, "claude code, anthropic's official cli") {
			cleanLines = append(cleanLines, claudeCodeSelfIDRe.ReplaceAllString(line, "You are a helpful coding assistant."))
			continue
		}

		cleanLines = append(cleanLines, line)
	}
	return strings.TrimSpace(strings.Join(cleanLines, "\n"))
}
