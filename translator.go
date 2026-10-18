package main

import (
	"encoding/base64"
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
//
// An audit caught that this only ever returned a single bool, which the
// caller fed straight into RecordEditResult as "isError" — a turn with NO
// edit at all (e.g. the model called Read to re-check context after a
// failed edit, a common and reasonable thing to do) returned false here
// exactly like a turn whose edit actually SUCCEEDED, and RecordEditResult
// treats "not isError" as "reset the streak to 0". So Edit(fail) -> Read
// -> Edit(fail) never reached the 3-failure threshold: the interposed
// Read silently erased the first failure's count instead of leaving it
// untouched. Returning whether an edit happened AT ALL, separately from
// whether it failed, lets the caller leave the streak alone on a non-edit
// turn instead of treating it as a reset.
func lastTurnHadFailedEdit(reqMap map[string]interface{}) (hadEdit bool, failed bool) {
	msgsRaw, ok := reqMap["messages"].([]interface{})
	if !ok || len(msgsRaw) < 2 {
		return false, false
	}
	last, ok := msgsRaw[len(msgsRaw)-1].(map[string]interface{})
	if !ok || last["role"] != "user" {
		return false, false
	}
	prev, ok := msgsRaw[len(msgsRaw)-2].(map[string]interface{})
	if !ok || prev["role"] != "assistant" {
		return false, false
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
		return false, false
	}

	// A Round 15 audit caught that "failed" was set true the moment ANY
	// edit in this turn had is_error:true, even when a SIBLING edit in
	// the same turn succeeded (e.g. a multi-file batch where only one
	// obscure file's edit failed) — that's forward progress, not a stuck
	// loop, but it still counted toward the same 3-strike streak as a
	// genuinely repeated failure on one file. Only report failure when
	// EVERY edit in the turn failed.
	successCount, failedCount := 0, 0
	if blocks, ok := last["content"].([]interface{}); ok {
		for _, bi := range blocks {
			b, ok := bi.(map[string]interface{})
			if !ok || b["type"] != "tool_result" {
				continue
			}
			tid, _ := b["tool_use_id"].(string)
			if !editToolIDs[tid] {
				continue
			}
			hadEdit = true
			if isErr, _ := b["is_error"].(bool); isErr {
				failedCount++
			} else {
				successCount++
			}
		}
	}
	failed = failedCount > 0 && successCount == 0
	return hadEdit, failed
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
		// A Round 15 audit caught that this is called again on a growing
		// combined parts list as consecutive same-role turns keep
		// merging (see the caller re-sealing an already-sealed previous
		// turn's parts) — target can already carry a user_note from an
		// EARLIER merge round, and sjson.Set here blindly overwrote it
		// with only the current round's note, discarding the first
		// round's text entirely. Concatenate instead of replacing.
		newNote := note.String()
		if existing := gjson.Get(target, "functionResponse.response.user_note").String(); existing != "" {
			newNote = existing + "\n" + newNote
		}
		target, _ = sjson.Set(target, "functionResponse.response.user_note", newNote)
	}
	fnResp[0] = target
	return fnResp, followUpImages
}

// decodeBase64Loose decodes a base64 string that may contain embedded
// whitespace (RFC 2045 line-wrapping at 76 chars, or stray \r\n/\n/spaces
// some clients leave in). base64.StdEncoding.DecodeString rejects any
// non-alphabet byte outright with CorruptInputError, silently dropping an
// otherwise well-formed document whose encoder happened to wrap lines.
func decodeBase64Loose(s string) ([]byte, error) {
	return base64.StdEncoding.DecodeString(stripBase64Whitespace(s))
}

// stripBase64Whitespace removes RFC 2045 line-wrapping whitespace (76-char
// line wraps, or stray \r\n/\n/spaces some clients leave in) from a base64
// string. decodeBase64Loose uses this before decoding locally; inlineData.data
// (sent to Gemini for it to decode) needs the same cleanup for the same
// reason — some clients produce valid, spec-legal line-wrapped base64 that a
// strict decoder on the receiving end rejects outright.
func stripBase64Whitespace(s string) string {
	return strings.Map(func(r rune) rune {
		switch r {
		case ' ', '\t', '\n', '\r':
			return -1
		}
		return r
	}, s)
}

// stripDataURIPrefix strips a leading "data:<mime>;base64," prefix if
// present, returning only the base64 payload after the comma. Anthropic's
// own API never puts the prefix in a source block's "data" field (just the
// raw base64), but some clients mistakenly pass the full data: URI through
// anyway, which breaks every base64 decoder downstream outright since the
// prefix text itself isn't valid base64.
func stripDataURIPrefix(s string) string {
	if !strings.HasPrefix(s, "data:") {
		return s
	}
	if idx := strings.Index(s, ";base64,"); idx != -1 {
		return s[idx+len(";base64,"):]
	}
	return s
}

// builtinAnthropicToolSchema returns the reference input_schema for one of
// Anthropic's server-side built-in tools (bash, text_editor, computer use —
// web_search is handled separately via native Gemini googleSearch grounding
// and never reaches this path). Anthropic's own clients declare these with
// only {"type": "<dated-version>", "name": "..."} and NO input_schema at
// all — Anthropic's server fills in the real parameters itself — but this
// gateway has no equivalent server-side knowledge, so without this, an
// empty input_schema fell into the generic empty-object placeholder and
// told Gemini the tool takes nothing but a synthetic "reason" string.
// Matches by prefix since Anthropic dates each version (bash_20250124,
// bash_20241022, ...); the parameter shapes have stayed stable across
// dates for each tool family so one schema per family is sufficient.
// isDigitsOnly reports whether s is non-empty and consists solely of ASCII
// digits — used to recognize Anthropic's dated-version suffix convention
// (e.g. "20250124") without matching an arbitrary longer suffix.
func isDigitsOnly(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

func builtinAnthropicToolSchema(toolType string) (string, bool) {
	switch {
	case strings.HasPrefix(toolType, "bash_"):
		// An audit caught "required" was missing entirely — without it,
		// the model frequently called bash() with no "command" at all
		// (restart is an alternative mode, correctly left optional).
		return `{"type":"object","properties":{"command":{"type":"string","description":"The bash command to run."},"restart":{"type":"boolean","description":"Set to true to restart the tool instead of running a command."}},"required":["command"]}`, true
	case strings.HasPrefix(toolType, "text_editor_"):
		return `{"type":"object","properties":{"command":{"type":"string","enum":["view","create","str_replace","insert","undo_edit"],"description":"The editor command to run."},"path":{"type":"string","description":"Absolute path to the file or directory."},"file_text":{"type":"string","description":"Required for create: the full content of the new file."},"old_str":{"type":"string","description":"Required for str_replace: the exact text to replace."},"new_str":{"type":"string","description":"Replacement text for str_replace, or the text to insert for insert."},"insert_line":{"type":"integer","description":"Required for insert: the line number after which to insert."},"view_range":{"type":"array","items":{"type":"integer"},"description":"Optional for view: [start_line, end_line]."}},"required":["command","path"]}`, true
	case strings.HasPrefix(toolType, "computer_") && isDigitsOnly(strings.TrimPrefix(toolType, "computer_")):
		// A Round 15 audit flagged that a bare "computer_" prefix would
		// also swallow a hypothetical future variant like
		// "computer_toolset_20260801" that splits computer-use into many
		// separate, individually-schemaed tools (screenshot, left_click,
		// ...) rather than this one monolithic action-dispatch shape —
		// routing those through this schema instead of their own would
		// give the model the wrong parameters entirely. Only match the
		// established dated-version convention this bash_/text_editor_
		// family actually uses ("computer_" + a pure numeric date stamp,
		// e.g. "computer_20250124"), the same way every other
		// known-real builtin type name in this switch is still just a
		// stable family prefix.
		return `{"type":"object","properties":{"action":{"type":"string","enum":["key","hold_key","type","cursor_position","mouse_move","left_mouse_down","left_mouse_up","left_click","left_click_drag","right_click","middle_click","double_click","triple_click","scroll","wait","screenshot"],"description":"The computer-use action to perform."},"coordinate":{"type":"array","items":{"type":"integer"},"description":"[x, y] pixel coordinate for click/move/drag actions."},"text":{"type":"string","description":"Text for key/type actions."},"duration":{"type":"integer","description":"Seconds to hold a key or wait."},"scroll_direction":{"type":"string","enum":["up","down","left","right"]},"scroll_amount":{"type":"integer"}},"required":["action"]}`, true
	}
	return "", false
}

// sanitizeGeminiFunctionName rewrites a tool name to satisfy Gemini's
// function-calling protobuf identifier regex (^[a-zA-Z_][a-zA-Z0-9_]*$, max
// 63 chars) — Anthropic's own looser `^[a-zA-Z0-9_-]{1,64}$` permits hyphens
// (common in MCP tool names like "mcp__brave-search__search") that Gemini
// rejects outright. Any character outside [a-zA-Z0-9_] becomes "_"; a name
// that would otherwise start with a digit gets a leading "_".
func sanitizeGeminiFunctionName(name string) string {
	var b strings.Builder
	for _, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' {
			b.WriteRune(r)
		} else {
			b.WriteByte('_')
		}
	}
	out := b.String()
	if out == "" {
		return "_"
	}
	if out[0] >= '0' && out[0] <= '9' {
		out = "_" + out
	}
	if len(out) > 63 {
		out = out[:63]
	}
	return out
}

// ConvertClaudeRequestToGemini converts an Anthropic Messages API payload into
// a Google Gemini streamGenerateContent payload with full sanitization and tiering.
func ConvertClaudeRequestToGemini(claudeJSON []byte, tenant TenantConfig, reqClass string, agentKey string, sessionID string, instanceID string) ([]byte, string, int, bool, map[string]bool, map[string]string, string, error) {
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
		return nil, "", 0, false, nil, nil, "", fmt.Errorf("invalid json: %w", err)
	}

	// 1. Dynamic Tiering & Model Effort Resolution
	modelName, budget := ResolveTieringAndEffort(reqMap, reqClass, agentKey)
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

	// 1. Initial AST Sanitization (Middle Truncation 64KB, role repair, system extraction)
	_, wasShrunk, prefillText := SanitizeMessagesAndGraph(reqMap)

	// Context Window Safety Truncation: runs AFTER initial sanitization so token budget
	// is measured against middle-truncated outputs and deduplicated reads, preventing
	// premature or unnecessary message dropping.
	if dropped, truncated := TruncateConversationToFit(reqMap, modelName, requestedMaxOutputHint); truncated {
		LogGateway("[ContextWindow] dropped %d oldest message(s) to fit %s's context window [%s]", dropped, modelName, agentKey)
		// Re-sanitize to repair role alternation and orphaned tool calls created by dropping intermediate turns
		if _, _, pt := SanitizeMessagesAndGraph(reqMap); pt != "" {
			prefillText = pt
		}
	}

	// 2. Loop-Guard Nudges (Read-Loop & Oscillation)
	hasModifyingTools := false
	isPlanMode := false
	if toolsArr, ok := reqMap["tools"].([]interface{}); ok {
		for _, t := range toolsArr {
			if tMap, isMap := t.(map[string]interface{}); isMap {
				rawName := fmt.Sprint(tMap["name"])
				name := strings.ToLower(rawName)
				// An audit caught this switch had its own narrower, stale
				// copy of exactly what server.go's isModifyingToolName
				// already fixed — missing "update" entirely, and no MCP
				// write/create/modify tool detection at all. A session
				// using only an MCP write tool (no native Edit/Write
				// declared) was wrongly classified as read-only, getting
				// the main agent's tight 10-reads-in-15-turns threshold
				// instead of the relaxed explore one. Reuse that same
				// function instead of keeping a second copy to drift out
				// of sync again.
				if isModifyingToolName(rawName) {
					hasModifyingTools = true
				}
				switch name {
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
	// An audit caught that the Plan-Mode window-clear only ever ran inside
	// RecordReadTurnWithRealProgress, called from server.go AFTER this
	// same request's Gemini response comes back — one step too late for
	// THIS check, which decides whether to nudge the OUTGOING request. On
	// the exact request that exits Plan Mode, the threshold below still
	// compared against the stale, un-cleared Plan-Mode-era window. Sync
	// it here first; RecordReadTurnWithRealProgress calling it again
	// later this same request is a harmless no-op.
	globalLoopGuard.SyncPlanMode(loopKey, isPlanMode)
	if level := globalLoopGuard.CheckAndResetReadStreak(loopKey, readThreshold); level > 0 {
		nudgeText := NudgeReadLoop
		if level >= 2 {
			nudgeText = NudgeReadLoopEscalated
		}
		InjectUserNudge(reqMap, nudgeText)
		LogGateway(".. read-loop window >= %d Read calls [%s] level=%d -> injecting stop-nudge", readThreshold, loopKey, level)
	}
	if globalLoopGuard.CheckAndResetOscillation(loopKey) {
		InjectUserNudge(reqMap, NudgeOscillation)
		LogGateway(".. tool-call oscillation [%s] -> injecting loop-nudge", loopKey)
	}
	if hadEdit, editFailed := lastTurnHadFailedEdit(reqMap); hadEdit {
		if editFailStreak := globalLoopGuard.RecordEditResult(loopKey, editFailed); editFailStreak >= 3 {
			InjectUserNudge(reqMap, NudgeEditFailureLoop)
			LogGateway(".. %d consecutive Edit failures [%s] -> injecting edit-failure-loop nudge", editFailStreak, loopKey)
		}
	}

	// 3. Final graph validation after any loop-guard nudges
	if _, didShrink, pt := SanitizeMessagesAndGraph(reqMap); didShrink {
		wasShrunk = true
		if pt != "" {
			prefillText = pt
		}
	}

	// Re-serialize sanitized request
	sanitizedBytes, err := json.Marshal(reqMap)
	if err != nil {
		return nil, "", 0, false, nil, nil, "", fmt.Errorf("re-marshal sanitized json: %w", err)
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
	// disable_parallel_tool_use was completely unhandled — Gemini frequently
	// returns multiple functionCall parts in one candidate regardless, so
	// without this a client that set the flag (expecting at most one
	// tool_use per turn, per Anthropic's own contract) silently got several
	// anyway. The actual enforcement (dropping any call past the first) is
	// in sse.go/server.go — this system instruction is a best-effort nudge
	// to stop Gemini from generating the extra calls in the first place.
	// An audit caught that the official Anthropic SDKs actually send this
	// nested inside tool_choice (`tool_choice: {type:"auto",
	// disable_parallel_tool_use:true}`), not as a top-level request field —
	// the original check only matched the latter, which real clients may
	// never even send. Check both.
	if gjson.GetBytes(sanitizedBytes, "disable_parallel_tool_use").Bool() || gjson.GetBytes(sanitizedBytes, "tool_choice.disable_parallel_tool_use").Bool() {
		systemPrompt = strings.TrimSpace(systemPrompt + "\n\nYou must call at most ONE tool in this response. Do not perform parallel/simultaneous tool calls.")
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
	// effectiveToolNames mirrors the SAME filtering the tools-declaration
	// loop further below applies (strip_search_tools, the synthetic
	// web_search/google_search marker) — computed here, before messages
	// are translated, specifically so the tool_use/tool_result replay
	// below can tell a tool that's merely DECLARED by the client apart
	// from one actually SENT to Gemini this request. See its use below.
	effectiveToolNames := make(map[string]bool)
	// disambiguatedToolName is the SINGLE source of truth for each
	// original tool name's final, Gemini-bound sanitized name, including
	// any collision suffix. An audit caught that history replay (right
	// below, Section 5) ran strictly BEFORE the tools-declaration loop
	// (Section 6, further down this file) where collision disambiguation
	// used to be computed — history emitted the RAW sanitized name with
	// no suffix, while the declaration got "_1" appended for the second
	// of two colliding names (e.g. "docker:run" and "docker_run" both
	// sanitize to "docker_run"), so turn 2+ replayed a functionCall.name
	// that matched NO declared function, 400ing. Computing the full
	// disambiguation map here, before anything else runs, and having
	// BOTH history replay and the declaration loop read from this same
	// map guarantees they can never disagree.
	disambiguatedToolName := make(map[string]string)
	claimedSanitizedNames := make(map[string]string) // sanitized name -> the first original name that claimed it
	earlyForcedToolName := ""
	toolChoiceType := ""
	if tc := root.Get("tool_choice"); tc.Exists() {
		toolChoiceType = tc.Get("type").String()
		if toolChoiceType == "tool" {
			earlyForcedToolName = tc.Get("name").String()
		}
	}
	// An audit caught that tool_choice:"any" (forced tool use, no specific
	// name) wasn't covered by the forcedToolName exemption above (which
	// only ever fires for tool_choice:"tool" naming ONE tool) — if every
	// tool the client declared happens to be a search tool
	// (Grep/Glob, the only two in DropSearchTools), StripSearchTools
	// removed BOTH, leaving zero declared tools for Gemini to honor a
	// forced "any" tool-use request with at all; Gemini just answers with
	// plain text instead, violating the client's explicit contract.
	// Detect this specific all-tools-would-be-stripped case up front and
	// suppress stripping entirely for this one request if so — only a
	// request where literally every declared tool is a search tool hits
	// this, so it never weakens the strip for any request with at least
	// one non-search tool available.
	keepSearchToolsForForcedAny := false
	if toolChoiceType == "any" {
		keepSearchToolsForForcedAny = true
		if toolsArr := root.Get("tools"); toolsArr.IsArray() {
			toolsArr.ForEach(func(_, t gjson.Result) bool {
				n := t.Get("name").String()
				normName := strings.ToLower(strings.ReplaceAll(n, "_", ""))
				if normName == "websearch" || normName == "googlesearch" {
					return true
				}
				if !DropSearchTools[n] {
					keepSearchToolsForForcedAny = false
					return false
				}
				return true
			})
		} else {
			keepSearchToolsForForcedAny = false
		}
	}
	if toolsArr := root.Get("tools"); toolsArr.IsArray() {
		toolsArr.ForEach(func(_, t gjson.Result) bool {
			n := t.Get("name").String()
			if n == "" {
				return true
			}
			declaredToolNames[n] = true
			sanitized := sanitizeGeminiFunctionName(n)
			if existingOrig, claimed := claimedSanitizedNames[sanitized]; claimed && existingOrig != n {
				// A Round 15 audit caught that this only ever tried ONE
				// alternate candidate — if that candidate itself happened
				// to already be claimed (e.g. a client declaring
				// "docker:run", "docker_run_1", AND "docker_run": the
				// first sanitizes to "docker_run" and claims it, the
				// second claims "docker_run_1" as-is with no collision at
				// all, then the third literally IS "docker_run" and
				// collides with the first — generating a suffixed
				// candidate that could coincide with the second tool's
				// already-claimed name depending on iteration order), the
				// single assignment below silently overwrote that OTHER
				// tool's claim, leaving two distinct Claude-side tool
				// names mapped to the identical sanitized Gemini name —
				// Gemini then rejects the whole request with a duplicate
				// function-declaration error. Keep incrementing the
				// suffix until an actually-free slot is found.
				base := sanitized
				for n2 := 1; ; n2++ {
					suffix := fmt.Sprintf("_%d", n2)
					candidateBase := base
					if len(candidateBase)+len(suffix) > 63 {
						candidateBase = candidateBase[:63-len(suffix)]
					}
					candidate := candidateBase + suffix
					if existingOrig2, claimed2 := claimedSanitizedNames[candidate]; !claimed2 || existingOrig2 == n {
						sanitized = candidate
						break
					}
				}
			}
			claimedSanitizedNames[sanitized] = n
			disambiguatedToolName[n] = sanitized
			normName := strings.ToLower(strings.ReplaceAll(n, "_", ""))
			if normName == "websearch" || normName == "googlesearch" {
				return true
			}
			if tenant.StripSearchTools && DropSearchTools[n] && n != earlyForcedToolName && !keepSearchToolsForForcedAny {
				return true
			}
			effectiveToolNames[n] = true
			return true
		})
	}
	messages := root.Get("messages")

	// An audit of Google's own equivalent agy.exe converter (reverse-
	// engineered, but this specific point is independently corroborated
	// by Anthropic's own public extended-thinking documentation, not just
	// the decompile) flagged that this gateway resends EVERY historical
	// "thinking" block on EVERY subsequent request — Anthropic's own spec
	// only requires preserving a thinking block for continuity within an
	// ACTIVE tool-use loop (the most recent assistant turn, whose
	// signature Google still needs to verify reasoning continuity against
	// a pending tool_result); once a turn is genuinely complete, resending
	// its old reasoning serves no purpose and just accumulates context
	// bloat over a long agentic session. Only the single most recent
	// assistant message's thinking is kept; every earlier one is dropped
	// (not placeholder'd — Gemini has no requirement that a thought be
	// replaced with anything, unlike the image/document role-mismatch
	// cases elsewhere in this file).
	lastAssistantIdx := -1
	if messages.IsArray() {
		msgArr := messages.Array()
		for i := len(msgArr) - 1; i >= 0; i-- {
			if msgArr[i].Get("role").String() == "assistant" {
				lastAssistantIdx = i
				break
			}
		}
	}

	if messages.IsArray() {
		contentIdx := 0
		var lastRole string

		messages.ForEach(func(msgIdxResult, msg gjson.Result) bool {
			role := msg.Get("role").String()
			geminiRole := "user"
			if role == "assistant" {
				geminiRole = "model"
			}
			isLastAssistantMsg := role == "assistant" && int(msgIdxResult.Int()) == lastAssistantIdx

			// Pre-pass: map tool_use IDs to function names
			if parts := msg.Get("content"); parts.IsArray() {
				parts.ForEach(func(_, p gjson.Result) bool {
					// A follow-up audit caught that this ran regardless of
					// the enclosing message's role, same blind spot
					// sanitizer.go's toolUsesSeen had (now fixed) — a
					// tool_use sitting in a role-inverted USER message
					// still got its id registered here, even though the
					// actual tool_use block itself degrades to plain text
					// below (role != "assistant" case) and never becomes a
					// real functionCall. A later tool_result for that same
					// id then found a non-empty fnName and emitted a real
					// functionResponse with no matching functionCall ever
					// sent — an orphaned response Gemini rejects.
					if p.Get("type").String() == "tool_use" && role == "assistant" {
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
						if role == "assistant" {
							// Gemini only allows text/thought/functionCall
							// parts in a "model"-role turn — inlineData there
							// (an imported conversation history, or a
							// multimodal subagent's own prior turn, can
							// legitimately contain an image on the assistant
							// side) gets rejected outright with "Invalid part
							// type for model role". Swap it for a text
							// placeholder instead of silently producing an
							// upstream 400.
							//
							// An audit caught that this fired unconditionally
							// for EVERY "image"/"document" block in an
							// assistant turn, including genuinely TEXT
							// documents (source.type == "text"/"content", or
							// base64 with a text-ish media_type like
							// text/plain or text/markdown) — those decode to
							// a plain "text" part below, which Gemini allows
							// fine in the model role; only the actual binary
							// cases (an image, or a base64 PDF/image) need
							// the placeholder swap. Replacing a generated
							// code file's full content with a one-line
							// placeholder destroyed the model's own prior
							// output from history for no reason.
							srcTypePeek := p.Get("source.type").String()
							mtPeek := strings.ToLower(p.Get("source.media_type").String())
							// An audit caught that SVG (image/svg+xml) was
							// never excluded here the way every other
							// image-pricing/routing site in this codebase
							// (server.go, context_window.go,
							// token_estimator.go, and this same file's own
							// base64-decode switch below) already excludes
							// it — SVG is XML text, not binary raster data,
							// and Gemini accepts it fine as a plain text
							// part in the model role. An assistant-authored
							// SVG diagram/artifact was being destroyed and
							// replaced with a one-line placeholder for no
							// reason, the exact same class of bug this
							// whole isBinary check exists to AVOID for
							// genuine text documents.
							isBinary := ptype == "image" && mtPeek != "image/svg+xml"
							if ptype == "document" && srcTypePeek == "base64" {
								effectiveMT := mtPeek
								if effectiveMT == "" {
									effectiveMT = "application/pdf"
								}
								isBinary = (effectiveMT == "application/pdf" || strings.HasPrefix(effectiveMT, "image/")) && effectiveMT != "image/svg+xml"
							}
							if isBinary {
								part := `{"text":""}`
								part, _ = sjson.Set(part, "text", "[Assistant output "+ptype+": "+mtPeek+"]")
								partsJSON = append(partsJSON, part)
								return true
							}
						}
						if ptype == "document" {
							// An audit caught that Anthropic's document
							// block's "title" and "context" fields (e.g. the
							// real filename, or a caller-supplied
							// description) were dropped entirely — Gemini
							// then sees only raw inlineData bytes with no
							// name or description attached, so "analyze
							// report_2026.pdf" has nothing for the model to
							// connect to the attachment it's looking at.
							title := p.Get("title").String()
							docContext := p.Get("context").String()
							if title != "" || docContext != "" {
								label := "[Attached Document"
								if title != "" {
									label += ": \"" + title + "\""
								}
								if docContext != "" {
									label += " - Context: " + docContext
								}
								label += "]"
								part := `{"text":""}`
								part, _ = sjson.Set(part, "text", label)
								partsJSON = append(partsJSON, part)
							}
						}
						src := p.Get("source")
						switch src.Get("type").String() {
						case "base64":
							// An audit caught this comparison ran without
							// strings.ToLower(): a client sending
							// "APPLICATION/PDF" or "IMAGE/PNG" (valid per
							// MIME's case-insensitivity) missed both checks
							// and fell into the text-decode branch below,
							// which wraps the still-binary bytes in a Go
							// string and sends them as a "text" part —
							// invalid UTF-8 that Gemini rejects or garbles.
							mediaType := strings.ToLower(src.Get("media_type").String())
							if mediaType == "" && ptype == "document" {
								// An audit caught context_window.go already
								// defaults a missing media_type to
								// "application/pdf" for TOKEN COUNTING
								// purposes, but this is the separate
								// function that translates the actual
								// bytes — without the same default, a
								// document block with no media_type fell
								// into the text-decode branch below,
								// decoding binary PDF bytes as if UTF-8 and
								// sending invalid text to Gemini (400).
								mediaType = "application/pdf"
							} else if mediaType == "" && ptype == "image" {
								// A follow-up audit caught the same gap for
								// "image" blocks specifically — this default
								// only ever covered "document", so a
								// no-media_type image block ALSO fell into
								// the text-decode branch below, producing the
								// exact same invalid-UTF-8-as-text failure.
								// PNG is the most common lossless screenshot
								// format in practice (Claude Code's own
								// screenshot tooling), so it's the more
								// reasonable default than erroring outright.
								mediaType = "image/png"
							}
							// An audit caught two gaps here: (1) a client
							// sending the FULL data URI ("data:image/png;
							// base64,AAAA...") in the "data" field instead
							// of just the base64 payload breaks any base64
							// decoder outright — strip that prefix if
							// present; (2) image/svg+xml matched
							// strings.HasPrefix(mediaType, "image/") and
							// got routed to inlineData, but SVG is XML
							// TEXT, not a real raster/vector type Gemini's
							// inlineData accepts — it rejects it with
							// "Invalid mimeType: image/svg+xml". Decode it
							// as text instead, same as any other text-ish
							// base64 document.
							b64Data := stripDataURIPrefix(src.Get("data").String())
							if b64Data != "" {
								if mediaType != "image/svg+xml" && (mediaType == "application/pdf" || strings.HasPrefix(mediaType, "image/")) {
									part := `{"inlineData":{"mimeType":"","data":""}}`
									part, _ = sjson.Set(part, "inlineData.mimeType", mediaType)
									// An audit caught that decodeBase64Loose
									// (used for the text-document decode
									// branch just below) strips RFC 2045
									// line-wrapping whitespace before
									// decoding, but inlineData.data here was
									// sent through raw — a client that
									// MIME-wraps its base64 (a real, spec-
									// legal encoding some libraries produce)
									// passed literal newlines/spaces into
									// Gemini's own base64 decode, which some
									// strict decoders reject outright.
									part, _ = sjson.Set(part, "inlineData.data", stripBase64Whitespace(b64Data))
									partsJSON = append(partsJSON, part)
								} else {
									// Plain text, code, CSV, JSON base64 documents:
									// Gemini inlineData only accepts binary media types.
									// Decode base64 to UTF-8 text!
									if decoded, err := decodeBase64Loose(b64Data); err == nil && len(decoded) > 0 {
										part := `{"text":""}`
										part, _ = sjson.Set(part, "text", string(decoded))
										partsJSON = append(partsJSON, part)
									}
								}
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
						case "url":
							// An audit caught that a "url"-sourced
							// image/document block hit no case here at all
							// and vanished — if it was the message's only
							// content, the whole turn collapsed to the
							// sanitizer's "..." placeholder with no trace
							// an attachment ever existed. We don't fetch
							// arbitrary URLs ourselves, but surface what
							// was attached as text instead of silence.
							if u := src.Get("url").String(); u != "" {
								part := `{"text":""}`
								part, _ = sjson.Set(part, "text", "[Attached "+ptype+" by URL: "+u+"]")
								partsJSON = append(partsJSON, part)
							}
						case "content":
							// An audit caught that Anthropic's other text
							// document form — source.type == "content",
							// where "content" is either the literal text or
							// an array of {"type":"text","text":...} blocks
							// — hit no case here at all and was silently
							// dropped, same failure mode "url" above was
							// already fixed for.
							contentField := src.Get("content")
							var combined strings.Builder
							if contentField.IsArray() {
								contentField.ForEach(func(_, block gjson.Result) bool {
									if block.Get("type").String() == "text" {
										combined.WriteString(block.Get("text").String())
									} else if block.Type == gjson.String {
										combined.WriteString(block.String())
									}
									return true
								})
							} else if contentField.Type == gjson.String {
								combined.WriteString(contentField.String())
							}
							if combined.Len() > 0 {
								part := `{"text":""}`
								part, _ = sjson.Set(part, "text", combined.String())
								partsJSON = append(partsJSON, part)
							}
						}
					case "tool_use":
						name := p.Get("name").String()
						callID := p.Get("id").String()
						argsRaw := p.Get("input").Raw
						if argsRaw == "" || argsRaw == "null" {
							// gjson's .Raw for a JSON `null` value is the
							// literal string "null", not "" — a tool_use
							// block with "input": null (some clients omit
							// args entirely this way) passed straight
							// through to Gemini as "args": null, which
							// Gemini rejects with 400 since functionCall.args
							// must be an object.
							argsRaw = "{}"
						}
						// An audit caught that sse.go/server.go's own
						// outbound responses already run newly-generated
						// tool_use args through normalizeToolCallArgsToObject
						// (coercing a bare string/array into an object), but
						// this HISTORY-REPLAY path never did — history
						// containing a non-object input (a client-supplied
						// turn, or an older capture) sent e.g. `"npm test"`
						// straight through as functionCall.args, which
						// Gemini's protobuf Struct requirement rejects
						// outright (400: expected struct/object).
						argsRaw = normalizeToolCallArgsToObject(name, argsRaw)
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
						if role != "assistant" {
							// An audit caught that this unconditionally
							// emitted a functionCall regardless of the
							// enclosing message's role — Anthropic's own
							// spec only ever puts tool_use in an assistant
							// turn, but a malformed/replayed/third-party
							// client sending one inside a USER message
							// still produced "role":"user" with a
							// functionCall part, which Gemini's protobuf
							// validator rejects outright ("functionCall is
							// only supported for model role"), 400ing the
							// WHOLE request instead of just degrading this
							// one odd block. Fall back to a text summary
							// instead of a part type Gemini will never
							// accept in this role.
							part := `{"text":""}`
							part, _ = sjson.Set(part, "text", "[tool_use in unexpected role: "+name+"]")
							partsJSON = append(partsJSON, part)
							return true
						}
						if !effectiveToolNames[name] {
							// An audit caught that a tool_use for a tool
							// this request no longer declares to Gemini at
							// all — most commonly Grep/Glob under the
							// default strip_search_tools, used in an
							// EARLIER turn before being stripped from every
							// later turn's functionDeclarations — still got
							// replayed as a functionCall referencing that
							// undeclared name. Gemini validates every
							// functionCall.name in the conversation against
							// the CURRENT declared list and 400s the WHOLE
							// request the moment history contains one that
							// isn't there, permanently breaking any
							// multi-turn conversation that ever used a
							// since-stripped tool. Degrade to readable text
							// instead of emitting a functionCall Gemini will
							// reject outright.
							argsPreview := argsRaw
							if len(argsPreview) > 200 {
								argsPreview = argsPreview[:200] + "..."
							}
							part := `{"text":""}`
							part, _ = sjson.Set(part, "text", "[Called "+name+"("+argsPreview+")]")
							partsJSON = append(partsJSON, part)
							return true
						}
						part := `{"thoughtSignature":"","functionCall":{"name":"","args":{}}}`
						part, _ = sjson.Set(part, "thoughtSignature", sig)
						// An audit caught that sanitizeGeminiFunctionName
						// (added for the initial tools.functionDeclarations
						// list) was never applied when replaying a PRIOR
						// turn's tool_use back into history — a hyphenated
						// MCP tool name (e.g. "mcp__brave-search__search")
						// sent fine on turn 1 (declared tools get
						// sanitized), but turn 2+ resent the raw hyphenated
						// name here, and Gemini validates every
						// functionCall.name in the conversation against the
						// sanitized declared list, 400ing the whole request.
						// A follow-up audit caught that plain sanitization
						// alone isn't enough either: disambiguatedToolName
						// (computed once, up front) must be used so a
						// collision-disambiguated name ("docker_run_1")
						// matches exactly what the declaration loop sent —
						// falling back to plain sanitization only for a
						// tool_use whose name isn't in the CURRENT request's
						// declared tools at all (e.g. truncated-out history).
						callFnName := disambiguatedToolName[name]
						if callFnName == "" {
							callFnName = sanitizeGeminiFunctionName(name)
						}
						part, _ = sjson.Set(part, "functionCall.name", callFnName)
						if callID != "" {
							part, _ = sjson.Set(part, "functionCall.id", callID)
						}
						part, _ = sjson.SetRaw(part, "functionCall.args", argsRaw)
						partsJSON = append(partsJSON, part)

					case "tool_result":
						if role != "user" {
							// An audit caught the same role-inversion gap
							// as tool_use above, for the opposite
							// direction: Anthropic's spec only puts
							// tool_result in a USER turn, but a
							// malformed/replayed tool_result inside an
							// ASSISTANT message still unconditionally
							// produced a functionResponse part under
							// "role":"model" — which Gemini rejects
							// outright (functionResponse is only valid in
							// the user/function-response role), 400ing the
							// whole request.
							part := `{"text":""}`
							part, _ = sjson.Set(part, "text", "[tool_result in unexpected role]")
							partsJSON = append(partsJSON, part)
							return true
						}
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
							// A Round 15 audit caught that the random Go
							// map-iteration pick this used to fall back to
							// here was non-deterministic — the SAME
							// conversation history, resent unchanged on a
							// retry, could pick a DIFFERENT tool name each
							// time, changing the request's exact byte
							// content and invalidating Gemini's server-side
							// prefix cache for no functional reason.
							// root.Get("tools.0.name") below already covers
							// this exact case (declaredToolNames non-empty
							// implies at least one declared tool) and is
							// fully deterministic.
							if fnName == "" {
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
								// An audit caught that a tool_result.content
								// array of bare strings (e.g. ["line 1\n",
								// "line 2\n"], not typed {"type":"text",...}
								// blocks) has item.Get("type").String()=="" for
								// every element, matching no case below and
								// silently contributing nothing — 100% of a
								// tool's real output vanished for this shape.
								if item.Type == gjson.String {
									sb.WriteString(item.String())
									return true
								}
								switch item.Get("type").String() {
								case "text":
									sb.WriteString(item.Get("text").String())
								case "image", "document":
									// Previously dropped entirely (only "text"
									// blocks were read), silently losing any
									// screenshot/plot/PDF a tool returned.
									// An audit caught that title/context
									// metadata was handled for a top-level
									// message's document block but ignored
									// here — a tool returning a named/
									// described document lost that label,
									// leaving the model with no name to
									// connect its analysis to.
									if title := item.Get("title").String(); title != "" {
										sb.WriteString("[Attached Document: \"" + title + "\"]\n")
									}
									if docContext := item.Get("context").String(); docContext != "" {
										sb.WriteString("[Context: " + docContext + "]\n")
									}
									src := item.Get("source")
									switch src.Get("type").String() {
									case "base64":
										// Same case-sensitivity bug as the
										// top-level content-block path above:
										// without ToLower, "IMAGE/PNG" etc.
										// missed both checks and got wrapped
										// as invalid-UTF-8 "text".
										mediaType := strings.ToLower(src.Get("media_type").String())
										// Same two gaps as the top-level
										// content-block path: a full data:
										// URI instead of a raw base64 payload,
										// and image/svg+xml (XML text, not a
										// real inlineData-compatible type).
										data := stripDataURIPrefix(src.Get("data").String())
										if mediaType == "" && data != "" {
											// Same missing-media_type default as
											// the top-level content-block path
											// above, applied here too for
											// consistency — otherwise this
											// nested tool_result path silently
											// drops the attachment entirely
											// instead of a best-effort default.
											if item.Get("type").String() == "document" {
												mediaType = "application/pdf"
											} else {
												mediaType = "image/png"
											}
										}
										if mediaType != "" && data != "" {
											if mediaType != "image/svg+xml" && (mediaType == "application/pdf" || strings.HasPrefix(mediaType, "image/")) {
												imgPart := `{"inlineData":{"mimeType":"","data":""}}`
												imgPart, _ = sjson.Set(imgPart, "inlineData.mimeType", mediaType)
												imgPart, _ = sjson.Set(imgPart, "inlineData.data", stripBase64Whitespace(data))
												imageParts = append(imageParts, imgPart)
											} else {
												if decoded, err := decodeBase64Loose(data); err == nil {
													sb.Write(decoded)
												}
											}
										}
									case "text":
										if txt := src.Get("data").String(); txt != "" {
											sb.WriteString(txt)
										}
									case "url":
										// Same as the top-level content-block
										// path: a url-sourced tool_result
										// attachment silently vanished instead
										// of at least surfacing as text.
										if u := src.Get("url").String(); u != "" {
											sb.WriteString("[Attached by URL: " + u + "]")
										}
									case "content":
										// An audit caught that the top-level
										// message-content path got this case
										// added, but this separate, nested
										// tool_result switch didn't — a text
										// document with source.type=="content"
										// returned BY A TOOL (as opposed to one
										// the user attached directly) still
										// silently vanished here.
										contentField := src.Get("content")
										if contentField.IsArray() {
											contentField.ForEach(func(_, block gjson.Result) bool {
												if block.Get("type").String() == "text" {
													sb.WriteString(block.Get("text").String())
												} else if block.Type == gjson.String {
													sb.WriteString(block.String())
												}
												return true
											})
										} else if contentField.Type == gjson.String {
											sb.WriteString(contentField.String())
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
						if !effectiveToolNames[fnName] {
							// Same undeclared-tool gap as tool_use above,
							// mirrored for its paired result — a
							// functionResponse naming a tool no longer in
							// this request's functionDeclarations (most
							// commonly Grep/Glob stripped by
							// strip_search_tools) 400s the whole request the
							// same way an orphaned functionCall does.
							resultPreview := resText
							if len(resultPreview) > 200 {
								resultPreview = resultPreview[:200] + "..."
							}
							part := `{"text":""}`
							part, _ = sjson.Set(part, "text", "[Result of "+fnName+": "+resultPreview+"]")
							partsJSON = append(partsJSON, part)
							return true
						}
						part := `{"functionResponse":{"name":"","response":{}}}`
						// Same disambiguation-consistency gap as
						// functionCall.name above — fnName here is the
						// raw Claude-side name recovered from toolIDToName.
						responseFnName := disambiguatedToolName[fnName]
						if responseFnName == "" {
							responseFnName = sanitizeGeminiFunctionName(fnName)
						}
						part, _ = sjson.Set(part, "functionResponse.name", responseFnName)
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
							if !isLastAssistantMsg {
								// This thinking block belongs to a completed,
								// older assistant turn — not the one Google
								// might still need reasoning continuity for
								// against a pending tool_result. Drop it
								// silently (no placeholder needed; Gemini has
								// no requirement that a dropped thought be
								// replaced with anything) instead of resending
								// every historical thought on every request,
								// which accumulates real context bloat over a
								// long agentic session for zero benefit.
								return true
							}
							// Prefer the real signature the client echoed back
							// (now that sse.go actually sends one via
							// signature_delta) over the skip-validator bypass;
							// Google warns that permanently skipping the
							// validator can degrade reasoning quality on long
							// thinking chains.
							sig := p.Get("signature").String()
							if sig == "" || sig == "skip_thought_signature_validator" {
								// A follow-up audit caught a recoverable case:
								// in a thought->text->functionCall sequence
								// split across SSE chunks, the live stream
								// (sse.go) closes the thinking block with the
								// skip-validator placeholder before the
								// LATER functionCall's real thoughtSignature
								// ever arrives — but that real signature IS
								// cached (setToolSignature, sse.go) against
								// the functionCall's own tool_use id. On
								// REPLAY (this history-translation path, a
								// later turn), the sibling tool_use this
								// same assistant message carries is right
								// here in `content` — check it for a real
								// cached signature before giving up and
								// resending the same placeholder forever.
								// Unlike buffering the live stream itself
								// (assessed independently five times as too
								// risky — delays TTFT for every response),
								// this only runs once, on the next turn,
								// with no latency cost at all.
								{
									// An audit caught this was gated strictly
									// on sig == "skip_thought_signature_validator",
									// but the outer check above also matches
									// sig == "" — a historical turn with a
									// genuinely empty signature (not yet the
									// placeholder) skipped this recovery
									// attempt entirely and went straight to
									// forcing the placeholder below, even
									// though the exact same sibling-lookup
									// could have found a real cached signature
									// for it too. Reaching this block at all
									// already guarantees sig is one of these
									// two values, so just always attempt it.
									content.ForEach(func(_, sibling gjson.Result) bool {
										if sibling.Get("type").String() != "tool_use" {
											return true
										}
										if real := getToolSignature(sibling.Get("id").String()); real != "" {
											sig = real
											return false
										}
										return true
									})
								}
								if sig == "" {
									sig = "skip_thought_signature_validator"
								}
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
						if role != "assistant" || !isLastAssistantMsg {
							// An audit caught that the "only keep thinking
							// on the most recent assistant turn" rule
							// (applied to genuine "thinking" blocks
							// elsewhere in this switch) was never extended
							// to this case — every historical
							// redacted_thinking block kept getting resent
							// as a thought:true part on every subsequent
							// request. Unlike real thinking, there's no
							// genuine signature here to preserve either way
							// (always the same skip_thought_signature_validator
							// placeholder), so there's no reason to ever
							// treat an OLDER one as thought:true — degrade
							// straight to plain text.
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
	// An audit caught that tool names were passed to Gemini verbatim —
	// Anthropic permits hyphens (`^[a-zA-Z0-9_-]{1,64}$`), and MCP tools
	// commonly use them (e.g. "mcp__brave-search__search"), but Gemini's
	// function-calling protobuf enforces a stricter
	// `^[a-zA-Z_][a-zA-Z0-9_]*$` (no hyphens, max 63 chars), rejecting the
	// whole request with HTTP 400 the moment any declared tool has one.
	// reverseToolNameMap lets the response path (server.go, sse.go)
	// restore the original name Claude Code expects once Gemini invokes
	// the function by its sanitized one.
	reverseToolNameMap := make(map[string]string)
	sentToolNames := make(map[string]string) // original name -> sanitized (Gemini-bound) name
	// Collision disambiguation (two original names sanitizing to the same
	// string) is now decided once, up front — see disambiguatedToolName's
	// doc comment near the top of this function — and simply looked up
	// here, rather than recomputed independently.
	// An audit caught that tool_choice:{"type":"tool","name":"Grep"} with
	// strip_search_tools enabled (the default) silently did nothing: Grep
	// never reached sentToolNames (stripped below before that point), so
	// the tool_choice handling further down found no match and skipped
	// setting toolConfig entirely — Gemini fell back to its own AUTO mode
	// and answered with plain text, violating the client's explicit
	// forced-tool-use request. A client naming a tool via tool_choice has
	// already said it specifically needs that exact tool; honor it over
	// the strip_search_tools default instead of silently ignoring both.
	forcedToolName := ""
	if tc := root.Get("tool_choice"); tc.Get("type").String() == "tool" {
		forcedToolName = tc.Get("name").String()
	}
	if tools := root.Get("tools"); tools.IsArray() {
		var decls []string
		hasWebSearch := false

		tools.ForEach(func(_, t gjson.Result) bool {
			name := t.Get("name").String()
			if name == "" {
				// An audit caught that a client declaring a built-in
				// Anthropic tool by "type" alone (e.g. {"type":
				// "bash_20250124"}, with no "name" key at all — valid per
				// Anthropic's spec, which lets the server default it)
				// left name=="" here, and sanitizeGeminiFunctionName("")
				// produces the useless declaration name "_". Derive a
				// real name from the type string itself, stripped of its
				// dated-version suffix ("bash_20250124" -> "bash").
				if toolType := t.Get("type").String(); toolType != "" {
					name = strings.TrimRight(strings.TrimRightFunc(toolType, func(r rune) bool {
						return r >= '0' && r <= '9'
					}), "_")
					if name == "" {
						name = toolType
					}
				}
			}
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
			if tenant.StripSearchTools && DropSearchTools[name] && name != forcedToolName && !keepSearchToolsForForcedAny {
				return true
			}
			desc := t.Get("description").String()
			if name == "Read" {
				// See loopguard.go's readToolGuidance doc comment for why
				// this lives on the tool's own description rather than the
				// system prompt: it's static (same text every request, so
				// it doesn't disturb Gemini's prefix cache) and sits right
				// next to the schema the model is already attending to
				// when it decides how to call THIS tool.
				desc = strings.TrimRight(desc, "\n") + "\n\n" + readToolGuidance
			}
			rawSchema := t.Get("input_schema").Raw
			// An audit caught that Anthropic's server-side built-in tools
			// (bash_20250124, text_editor_20250728, computer_20250124, and
			// their earlier dated variants) are declared by the client
			// WITHOUT input_schema at all — Anthropic's own server already
			// knows their parameters — but this gateway had no equivalent
			// knowledge, so an empty input_schema fell straight into the
			// generic "no params, needsPlaceholder" path below, telling
			// Gemini the tool takes only a synthetic "reason" string. The
			// model then called e.g. bash(reason="...") with no "command"
			// at all, since that's the only parameter the schema we sent
			// ever gave it. Inject the real reference schema by the
			// declaration's own "type" field before falling through.
			if strings.TrimSpace(rawSchema) == "" || rawSchema == "null" || rawSchema == "{}" {
				if builtin, ok := builtinAnthropicToolSchema(t.Get("type").String()); ok {
					rawSchema = builtin
				}
			}
			schemaRaw := CleanJSONSchemaForAntigravity(name, rawSchema)
			needsPlaceholder := strings.TrimSpace(schemaRaw) == "" || schemaRaw == "null"
			if !needsPlaceholder {
				// An audit caught that a schema like "{}" or
				// "{\"properties\":{}}" — no "type" field at all, so
				// addEmptySchemaPlaceholder's own findPaths(jsonStr, "type")
				// never even sees it to inject the placeholder — fell
				// through this check too (TrimSpace("{}") is neither ""
				// nor "null"), reaching Gemini as a bare "parameters: {}"
				// with no type and no properties. Same real defect as the
				// one this check already exists for, just a different
				// raw string shape; catch it the same way.
				propsVal := gjson.Get(schemaRaw, "properties")
				propsEmpty := !propsVal.Exists() || (propsVal.IsObject() && len(propsVal.Map()) == 0)
				schemaType := gjson.Get(schemaRaw, "type").String()
				if propsEmpty && (schemaType == "" || schemaType == "object") {
					needsPlaceholder = true
				}
			}
			if needsPlaceholder {
				// Google rejects an OBJECT-type schema with empty properties
				// ("parameters.properties: should be non-empty for OBJECT
				// type") — route through the same placeholder mechanism
				// CleanJSONSchemaForAntigravity already uses for schemas
				// that end up empty after cleaning, instead of a raw
				// properties:{} that bypasses it entirely.
				schemaRaw, _ = addEmptySchemaPlaceholder(`{"type":"object","properties":{}}`, nil)
			}
			// Remember which tools got our synthetic "reason" placeholder
			// injected at the TOP LEVEL (the only way Gemini will accept
			// an argument-less tool's schema) so the caller can strip
			// that key back out of the model's actual function-call args
			// before handing them to Claude Code — a strict MCP tool
			// with additionalProperties: false rejects a "reason" field
			// it never declared.
			//
			// An audit caught that a blind substring search across the
			// WHOLE schema string false-positived whenever the SAME
			// placeholder text happened to appear anywhere else in the
			// schema too — e.g. a nested free-dict field (like
			// "metadata") that independently got its own synthetic
			// reason injected for ITS OWN emptiness. That flagged the
			// entire tool, and the caller unconditionally deletes the
			// TOP-LEVEL "reason" key whenever flagged — destroying a
			// genuinely real, user-declared top-level "reason" argument
			// that tool also happened to have, unrelated to the nested
			// placeholder. Check the exact top-level path instead of a
			// substring search across the whole schema.
			if gjson.Get(schemaRaw, "properties.reason.description").String() == placeholderReasonDescription {
				syntheticReasonTools[name] = true
			}

			// Reuse the single disambiguation decision computed up front
			// (before message history was translated) rather than
			// recomputing it independently here — the two MUST agree, or
			// history replay and this declaration diverge on exactly the
			// collision case this mechanism exists to handle.
			sanitizedName, ok := disambiguatedToolName[name]
			if !ok {
				sanitizedName = sanitizeGeminiFunctionName(name)
			}
			if sanitizedName != name {
				reverseToolNameMap[sanitizedName] = name
			}
			decl := `{"name":"","description":"","parameters":{}}`
			decl, _ = sjson.Set(decl, "name", sanitizedName)
			decl, _ = sjson.Set(decl, "description", desc)
			decl, _ = sjson.SetRaw(decl, "parameters", schemaRaw)
			decls = append(decls, decl)
			sentToolNames[name] = sanitizedName
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
					if name := tc.Get("name").String(); name != "" {
						if sanitized, ok := sentToolNames[name]; ok {
							// allowedFunctionNames must reference the
							// SANITIZED (Gemini-bound) name, not the
							// original Claude-side one, or Gemini 400s
							// with "allowed_function_names contains X
							// which is not in function_declarations" the
							// moment the original name had a hyphen.
							out, _ = sjson.Set(out, "toolConfig.functionCallingConfig.mode", "ANY")
							out, _ = sjson.Set(out, "toolConfig.functionCallingConfig.allowedFunctionNames.0", sanitized)
						}
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
	// An audit caught that thinkingConfig was injected unconditionally,
	// with no check against the model catalog's own Thinking capability
	// flag — a 3rd-party model routed through CloudCode's native-model
	// passthrough (tiering.go's "Explicit native models preservation"
	// case, e.g. "gpt-oss-120b-medium") that genuinely doesn't support
	// thinking got thinkingConfig anyway, and Google rejects that outright
	// with "Thinking is not supported for this model". Only known-false
	// (hasMeta && !modelMeta.Thinking) skips it — an unrecognized model
	// still gets the existing best-effort behavior, same as before.
	if !hasMeta || modelMeta.Thinking {
		// An audit caught the MinThinkingBudget floor below only applied
		// inside the budget>0 branch — a model whose thinking genuinely
		// can't be disabled at all (e.g. Gemini 2.5 Pro has a non-zero
		// floor) still got an explicit thinkingBudget:0 whenever budget
		// resolved to exactly 0 (the standard Anthropic "thinking":
		// {"type":"disabled"} request, or any tiering decision landing
		// on 0), which Google rejects with "Thinking is not supported
		// for this model" / INVALID_ARGUMENT just the same as sending no
		// thinkingConfig at all would for a model that requires it.
		// Apply the floor whenever a non-negative budget is about to be
		// sent, not just a strictly positive one.
		if budget >= 0 && hasMeta && modelMeta.MinThinkingBudget > 0 && budget < modelMeta.MinThinkingBudget {
			budget = modelMeta.MinThinkingBudget
		}
		out, _ = sjson.Set(out, "generationConfig.thinkingConfig.includeThoughts", budget != 0)
		if budget < 0 {
			// -1 enables ModeAuto unbracketed
			out, _ = sjson.Set(out, "generationConfig.thinkingConfig.thinkingBudget", -1)
		} else {
			out, _ = sjson.Set(out, "generationConfig.thinkingConfig.thinkingBudget", budget)
		}
	}

	// 6.5 Carry over sampling hyperparameters: previously dropped entirely,
	// so a client asking for temperature:0 (deterministic code generation)
	// silently got Gemini's own default (commonly ~1.0) instead.
	if temp := root.Get("temperature"); temp.Exists() {
		t := temp.Float()
		if budget != 0 && t != 1.0 {
			// An audit caught that Anthropic's own spec requires
			// temperature == 1.0 whenever thinking is enabled (any other
			// value is rejected outright on the real API) — passing a
			// client's temperature:0 (the common default for
			// deterministic code generation) straight through here, on
			// top of a thinking budget, measured live to degrade Gemini
			// 2.5/3.8's reasoning into repetitive thought loops instead of
			// erroring, a much harder failure to notice and debug.
			t = 1.0
		}
		out, _ = sjson.Set(out, "generationConfig.temperature", t)
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
	//
	// A Round 15 audit caught this only ever read "output_config.format" —
	// some official SDK call shapes (e.g. client.messages.parse()) send
	// the format at the request's TOP level as "output_format" instead.
	// Fall back to that root field when the nested one is absent, rather
	// than silently treating the request as unstructured text.
	fmtNode := root.Get("output_config.format")
	if !fmtNode.Exists() {
		fmtNode = root.Get("output_format")
	}
	if fmtNode.Exists() {
		switch fmtNode.Get("type").String() {
		case "json_schema":
			out, _ = sjson.Set(out, "generationConfig.responseMimeType", "application/json")
			if schemaNode := fmtNode.Get("schema"); schemaNode.Exists() {
				cleanedSchema := CleanJSONSchemaForGemini(schemaNode.Raw)
				out, _ = sjson.SetRaw(out, "generationConfig.responseSchema", cleanedSchema)
			}
		case "json", "json_object":
			// A Round 15 audit caught that a non-strict free-form JSON
			// request (no "schema" key, just {"type":"json"} — "any valid
			// JSON, no fixed shape required") never matched the
			// json_schema-only check above, so the model was never told
			// to actually emit JSON and commonly wrapped its answer in a
			// markdown code fence instead of raw JSON.
			out, _ = sjson.Set(out, "generationConfig.responseMimeType", "application/json")
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

	return []byte(out), modelName, budget, wasShrunk, syntheticReasonTools, reverseToolNameMap, prefillText, nil
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
