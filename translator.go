package main

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

var DropSearchTools = map[string]bool{
	"Grep": true,
	"Glob": true,
}

// ConvertClaudeRequestToGemini converts an Anthropic Messages API payload into
// a Google Gemini streamGenerateContent payload with full sanitization and tiering.
func ConvertClaudeRequestToGemini(claudeJSON []byte, tenant TenantConfig, reqClass string, agentKey string) ([]byte, string, int, bool, error) {
	// Parse as generic map for in-place AST transformations
	var reqMap map[string]interface{}
	if err := json.Unmarshal(claudeJSON, &reqMap); err != nil {
		return nil, "", 0, false, fmt.Errorf("invalid json: %w", err)
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
	}
	if dropped, truncated := TruncateConversationToFit(reqMap, modelName, requestedMaxOutputHint); truncated {
		LogGateway("[ContextWindow] dropped %d oldest message(s) to fit %s's context window [%s]", dropped, modelName, agentKey)
	}

	// 2. Loop-Guard Nudges (Read-Loop & Oscillation)
	if globalLoopGuard.CheckAndResetReadStreak(agentKey) {
		InjectUserNudge(reqMap, NudgeReadLoop)
		LogGateway(".. read-loop window >= %d Read calls [%s] -> injecting stop-nudge", ReadWindowThreshold, agentKey)
	}
	if globalLoopGuard.CheckAndResetOscillation(agentKey) {
		InjectUserNudge(reqMap, NudgeOscillation)
		LogGateway(".. tool-call oscillation [%s] -> injecting loop-nudge", agentKey)
	}

	// 3. Full AST Sanitization (Middle Truncation 64KB, Non-empty sentinels, role repair, tool pairing)
	_, wasShrunk := SanitizeMessagesAndGraph(reqMap)

	// Re-serialize sanitized request
	sanitizedBytes, err := json.Marshal(reqMap)
	if err != nil {
		return nil, "", 0, false, fmt.Errorf("re-marshal sanitized json: %w", err)
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
		out, _ = sjson.Set(out, "system_instruction.role", "user")
		out, _ = sjson.Set(out, "system_instruction.parts.0.text", systemPrompt)
	}

	// 5. Messages Translation
	toolIDToName := make(map[string]string)
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
					case "image":
						src := p.Get("source")
						if src.Get("type").String() == "base64" {
							mediaType := src.Get("media_type").String()
							b64Data := src.Get("data").String()
							if b64Data != "" {
								part := `{"inlineData":{"mimeType":"","data":""}}`
								part, _ = sjson.Set(part, "inlineData.mimeType", mediaType)
								part, _ = sjson.Set(part, "inlineData.data", b64Data)
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
						part := `{"thoughtSignature":"skip_thought_signature_validator","functionCall":{"name":"","args":{}}}`
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
							fnName = "tool"
						}

						resText := ""
						c := p.Get("content")
						if c.IsArray() {
							var sb strings.Builder
							c.ForEach(func(_, item gjson.Result) bool {
								if item.Get("type").String() == "text" {
									sb.WriteString(item.Get("text").String())
								}
								return true
							})
							resText = sb.String()
						} else if c.Exists() {
							resText = c.String()
						}

						part := `{"functionResponse":{"name":"","response":{"result":""}}}`
						part, _ = sjson.Set(part, "functionResponse.name", fnName)
						if toolUseID != "" {
							part, _ = sjson.Set(part, "functionResponse.id", toolUseID)
						}
						part, _ = sjson.Set(part, "functionResponse.response.result", resText)
						partsJSON = append(partsJSON, part)

					case "thinking":
						thoughtTxt := p.Get("thinking").String()
						if thoughtTxt != "" {
							part := `{"thought":true,"thoughtSignature":"skip_thought_signature_validator","text":""}`
							part, _ = sjson.Set(part, "text", thoughtTxt)
							partsJSON = append(partsJSON, part)
						}
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

			if geminiRole == lastRole && contentIdx > 0 {
				targetIdx := contentIdx - 1
				for _, pjson := range partsJSON {
					out, _ = sjson.SetRaw(out, fmt.Sprintf("contents.%d.parts.-1", targetIdx), pjson)
				}
			} else {
				out, _ = sjson.Set(out, fmt.Sprintf("contents.%d.role", contentIdx), geminiRole)
				for _, pjson := range partsJSON {
					out, _ = sjson.SetRaw(out, fmt.Sprintf("contents.%d.parts.-1", contentIdx), pjson)
				}
				lastRole = geminiRole
				contentIdx++
			}

			return true
		})
	}

	// 6. Tools Declaration
	if tools := root.Get("tools"); tools.IsArray() {
		var decls []string
		hasWebSearch := false

		tools.ForEach(func(_, t gjson.Result) bool {
			name := t.Get("name").String()
			if strings.EqualFold(name, "WebSearch") || strings.EqualFold(name, "GoogleSearch") {
				hasWebSearch = true
			}
			if tenant.StripSearchTools && DropSearchTools[name] {
				return true
			}
			desc := t.Get("description").String()
			schemaRaw := CleanJSONSchemaForAntigravity(t.Get("input_schema").Raw)
			if strings.TrimSpace(schemaRaw) == "" || schemaRaw == "null" {
				schemaRaw = `{"type":"object","properties":{}}`
			}

			decl := `{"name":"","description":"","parameters":{}}`
			decl, _ = sjson.Set(decl, "name", name)
			decl, _ = sjson.Set(decl, "description", desc)
			decl, _ = sjson.SetRaw(decl, "parameters", schemaRaw)
			decls = append(decls, decl)
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
		} else if hasWebSearch {
			out, _ = sjson.SetRaw(out, "tools.0.googleSearch", "{}")
		}
	}

	// 7. Thinking & Output Tokens Configuration with Dynamic Model Limits
	modelMeta, hasMeta := globalModelCatalog.GetModel(modelName)

	if budget != 0 {
		out, _ = sjson.Set(out, "generationConfig.thinkingConfig.includeThoughts", true)
		if budget > 0 {
			if hasMeta && modelMeta.MinThinkingBudget > 0 && budget < modelMeta.MinThinkingBudget {
				budget = modelMeta.MinThinkingBudget
			}
			out, _ = sjson.Set(out, "generationConfig.thinkingConfig.thinkingBudget", budget)
		} else {
			// -1 enables ModeAuto unbracketed
			out, _ = sjson.Set(out, "generationConfig.thinkingConfig.thinkingBudget", -1)
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

	return []byte(out), modelName, budget, wasShrunk, nil
}

// ScrubBillingHeader strips out fingerprint headers and billing marks that trigger Google 429/Filter:
// 1. x-anthropic-billing-header
// 2. Canonical Claude Code self-identification preambles that Google CloudCode actively filters
// 3. Environmental noise (gitStatus, Recent commits, fast mode tags) to reduce prompt tokens
func ScrubBillingHeader(txt string) string {
	var cleanLines []string
	skipSection := false

	for _, line := range strings.Split(txt, "\n") {
		trimmed := strings.TrimSpace(line)
		lower := strings.ToLower(trimmed)

		// Filter billing header marks
		if strings.Contains(line, "x-anthropic-billing-header") {
			continue
		}
		if strings.Contains(line, "Claude Agent SDK") || strings.Contains(line, "built on Anthropic") {
			cleanLines = append(cleanLines, "You are a helpful coding assistant.")
			continue
		}

		// Filter heavy repetitive environment sections
		if trimmed == "# Environment" {
			skipSection = true
			continue
		}
		if skipSection {
			if strings.HasPrefix(trimmed, "# ") {
				skipSection = false
			} else {
				continue
			}
		}

		// Strip git status and fast mode noise lines
		if strings.HasPrefix(trimmed, "gitStatus:") ||
			strings.HasPrefix(trimmed, "Recent commits:") ||
			strings.HasPrefix(trimmed, "<fast_mode_info>") ||
			strings.HasPrefix(trimmed, "</fast_mode_info>") ||
			strings.Contains(trimmed, "git status at the start of the conversation") {
			continue
		}

		// Normalize self-identification
		if strings.Contains(lower, "you are claude code, anthropic's official cli") {
			cleanLines = append(cleanLines, "You are a helpful coding assistant.")
			continue
		}

		cleanLines = append(cleanLines, line)
	}
	return strings.TrimSpace(strings.Join(cleanLines, "\n"))
}
