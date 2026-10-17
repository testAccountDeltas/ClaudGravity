package main

import "fmt"

// Rule is one named invariant checked against every captured record
// independently. Every rule below is a literal regression test for a
// specific incident from the 2026-10-17/18 session (see
// AGENT_HANDOFF_NOTES.md's "live production breaks" list) or from one of
// the two audits that session verified — not a speculative new check.
// Add a new Rule here every time a future incident identifies a new
// invariant, so the next occurrence of the same class of bug is caught by
// running speccheck, not another manual investigation.
type Rule struct {
	Name  string
	Check func(rec CaptureRecord) []Violation
}

var AllRules = []Rule{
	{"RoleAlternation", checkRoleAlternation},
	{"ToolCallResponseParity", checkToolCallResponseParity},
	{"ContentNotCollapsed", checkContentNotCollapsed},
	{"RequiredResponseHeaders", checkRequiredResponseHeaders},
	{"SchemaKeywordAllowlist", checkSchemaKeywordAllowlist},
	{"TokenAccountingSanity", checkTokenAccountingSanity},
	{"SSEEventSequence", checkSSEEventSequence},
}

func violation(rec CaptureRecord, detail string) Violation {
	return Violation{RID: rec.rid(), Timestamp: rec.timestamp(), Detail: detail}
}

// asArray is a small helper since every leg's body decodes to
// map[string]interface{}/[]interface{} via encoding/json, never a
// concrete named struct — this file intentionally has zero dependency on
// the gateway's own types.
func asArray(v interface{}) []interface{} {
	a, _ := v.([]interface{})
	return a
}

func asMap(v interface{}) map[string]interface{} {
	m, _ := v.(map[string]interface{})
	return m
}

func asString(v interface{}) string {
	s, _ := v.(string)
	return s
}

// --- RoleAlternation --------------------------------------------------
// Gemini's contents[] must never repeat the same role two turns in a row.
// Caught live: loopguard.go's InjectUserNudge appending a second "user"
// turn when its content-type switch didn't recognize the type
// SanitizeMessagesAndGraph had already normalized the last turn's content
// to (fixed in 8ef075f).
func checkRoleAlternation(rec CaptureRecord) []Violation {
	body := asMap(rec.snapshotBody("gemini_request"))
	if body == nil {
		return nil
	}
	contents := asArray(body["contents"])
	var out []Violation
	prevRole := ""
	for i, c := range contents {
		turn := asMap(c)
		role := asString(turn["role"])
		if role == "" {
			continue
		}
		if role == prevRole {
			out = append(out, violation(rec, fmt.Sprintf("contents[%d] repeats role %q immediately after contents[%d]", i, role, i-1)))
		}
		prevRole = role
	}
	return out
}

// --- ToolCallResponseParity --------------------------------------------
// Every functionCall in turn N must be answered by exactly one
// functionResponse in turn N+1. Caught live: BUG-R02, a delayed subagent
// tool_result desynchronizing the function-call/response count Gemini
// requires strict adjacency for.
func checkToolCallResponseParity(rec CaptureRecord) []Violation {
	body := asMap(rec.snapshotBody("gemini_request"))
	if body == nil {
		return nil
	}
	contents := asArray(body["contents"])
	var out []Violation
	for i := 0; i < len(contents); i++ {
		turn := asMap(contents[i])
		if asString(turn["role"]) != "model" {
			continue
		}
		calls := 0
		for _, p := range asArray(turn["parts"]) {
			if asMap(asMap(p)["functionCall"]) != nil {
				calls++
			}
		}
		if calls == 0 {
			continue
		}
		responses := 0
		if i+1 < len(contents) {
			next := asMap(contents[i+1])
			for _, p := range asArray(next["parts"]) {
				if asMap(asMap(p)["functionResponse"]) != nil {
					responses++
				}
			}
		}
		if responses != calls {
			out = append(out, violation(rec, fmt.Sprintf("contents[%d] (model) has %d functionCall(s) but contents[%d] has %d functionResponse(s)", i, calls, i+1, responses)))
		}
	}
	return out
}

// --- ContentNotCollapsed -----------------------------------------------
// The exact signature of this session's two live incidents: a message's
// content silently replaced with the bare "..." sentinel while its
// neighbors have real (redacted-but-present) content. Checked on the
// CLAUDE-facing request leg, since that's where a human would notice the
// user's own words went missing.
func checkContentNotCollapsed(rec CaptureRecord) []Violation {
	body := asMap(rec.snapshotBody("claude_request"))
	if body == nil {
		return nil
	}
	msgs := asArray(body["messages"])
	if len(msgs) < 2 {
		return nil
	}
	collapsedCount := 0
	totalCount := 0
	for _, m := range msgs {
		msg := asMap(m)
		blocks := asArray(msg["content"])
		totalCount++
		if len(blocks) == 1 {
			b := asMap(blocks[0])
			if asString(b["type"]) == "text" && asString(b["text"]) == "..." {
				collapsedCount++
			}
		}
	}
	// One or two legitimate "..." sentinels (a genuinely empty turn) are
	// normal and intentional (sanitizer.go's own fallback) — the signature
	// of the actual bug is EVERY turn collapsing at once.
	if totalCount >= 2 && collapsedCount == totalCount {
		return []Violation{violation(rec, fmt.Sprintf("all %d message(s) in this request collapsed to the \"...\" placeholder", totalCount))}
	}
	return nil
}

// --- RequiredResponseHeaders --------------------------------------------
// Claude Code's own backoff/countdown reads these headers. Caught live:
// writeAnthropicError not setting anthropic-ratelimit-unified-reset on a
// 429 (fixed in ea9e5c6).
func checkRequiredResponseHeaders(rec CaptureRecord) []Violation {
	headers := rec.snapshotHeaders("claude_response")
	if headers == nil {
		return nil
	}
	body := asMap(rec.snapshotBody("claude_response"))
	status := 200
	if body != nil {
		if t, ok := body["type"].(string); ok && t == "error" {
			status = 0 // unknown status for a bare error body capture; skip
		}
	}
	if status == 0 {
		return nil
	}
	has := func(name string) bool {
		_, ok := headers[name]
		if ok {
			return true
		}
		for k := range headers {
			if equalFoldASCII(k, name) {
				return true
			}
		}
		return false
	}
	var out []Violation
	for _, want := range []string{
		"anthropic-ratelimit-requests-limit",
		"anthropic-ratelimit-requests-remaining",
		"anthropic-ratelimit-tokens-limit",
		"anthropic-ratelimit-tokens-remaining",
	} {
		if !has(want) {
			out = append(out, violation(rec, fmt.Sprintf("response missing expected header %q", want)))
		}
	}
	return out
}

func equalFoldASCII(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		ca, cb := a[i], b[i]
		if ca >= 'A' && ca <= 'Z' {
			ca += 'a' - 'A'
		}
		if cb >= 'A' && cb <= 'Z' {
			cb += 'a' - 'A'
		}
		if ca != cb {
			return false
		}
	}
	return true
}

// --- SchemaKeywordAllowlist ----------------------------------------------
// Gemini's OpenAPI-3.0 Protobuf schema validator rejects any key outside
// a known small subset. This is an INDEPENDENT re-check of
// gemini_schema.go's own stripping list, from outside that code — it
// catches a future code change that silently stops stripping one of
// these, which the unit-test-free translator currently has no other
// guard against.
var geminiAllowedSchemaKeys = map[string]bool{
	"type": true, "description": true, "properties": true, "required": true,
	"items": true, "enum": true, "nullable": true, "format": true,
}

func checkSchemaKeywordAllowlist(rec CaptureRecord) []Violation {
	body := asMap(rec.snapshotBody("gemini_request"))
	if body == nil {
		return nil
	}
	var out []Violation
	for _, t := range asArray(body["tools"]) {
		tool := asMap(t)
		decls := asArray(tool["functionDeclarations"])
		for _, d := range decls {
			decl := asMap(d)
			name := asString(decl["name"])
			params := asMap(decl["parameters"])
			walkSchemaKeys(params, name, &out, rec)
		}
	}
	return out
}

func walkSchemaKeys(node map[string]interface{}, toolName string, out *[]Violation, rec CaptureRecord) {
	if node == nil {
		return
	}
	for k, v := range node {
		if !geminiAllowedSchemaKeys[k] {
			*out = append(*out, violation(rec, fmt.Sprintf("tool %q schema has unsupported key %q — gemini_schema.go's stripping list may have regressed", toolName, k)))
		}
		switch k {
		case "properties":
			for _, prop := range asMap(v) {
				walkSchemaKeys(asMap(prop), toolName, out, rec)
			}
		case "items":
			walkSchemaKeys(asMap(v), toolName, out, rec)
		}
	}
}

// --- TokenAccountingSanity -----------------------------------------------
// Caught live tonight (indirectly): token-count mismatches between what
// the client is told and what Gemini actually billed are exactly the kind
// of thing that's invisible until a conversation quietly misbehaves.
func checkTokenAccountingSanity(rec CaptureRecord) []Violation {
	respBody := asMap(rec.snapshotBody("claude_response"))
	geminiBody := asMap(rec.snapshotBody("gemini_response"))
	if respBody == nil || geminiBody == nil {
		return nil
	}
	usage := asMap(respBody["usage"])
	if usage == nil {
		return nil
	}
	claudeIn, _ := usage["input_tokens"].(float64)
	cacheRead, _ := usage["cache_read_input_tokens"].(float64)
	cacheCreate, _ := usage["cache_creation_input_tokens"].(float64)
	claudeTotal := claudeIn + cacheRead + cacheCreate

	usageMeta := asMap(geminiBody["usageMetadata"])
	if usageMeta == nil {
		return nil
	}
	geminiPrompt, ok := usageMeta["promptTokenCount"].(float64)
	if !ok || geminiPrompt == 0 {
		return nil
	}
	delta := claudeTotal - geminiPrompt
	if delta < 0 {
		delta = -delta
	}
	if delta > geminiPrompt*0.10+50 {
		return []Violation{violation(rec, fmt.Sprintf("claude-side input accounting (%.0f) diverges from Gemini's own promptTokenCount (%.0f) by more than 10%%", claudeTotal, geminiPrompt))}
	}
	return nil
}

// --- SSEEventSequence -----------------------------------------------------
// Streaming captures must follow the documented Anthropic SSE event
// sequence. A gap or out-of-order index here is exactly what an official
// SDK's own strict parser will reject.
func checkSSEEventSequence(rec CaptureRecord) []Violation {
	respLeg := asMap(rec.snapshotBody("claude_response"))
	// Streaming responses are recorded as {"events": [...]} at the top of
	// the claude_response leg (see speccapture.go's Finish) rather than
	// under "body" — reach for that shape directly.
	legRaw, ok := rec["claude_response"].(map[string]interface{})
	if !ok {
		return nil
	}
	eventsRaw, ok := legRaw["events"].([]interface{})
	if !ok {
		// Non-streaming capture (a plain "body") — nothing for this rule.
		_ = respLeg
		return nil
	}
	if len(eventsRaw) == 0 {
		return nil
	}
	first := asString(asMap(eventsRaw[0])["event"])
	last := asString(asMap(eventsRaw[len(eventsRaw)-1])["event"])
	var out []Violation
	if first != "message_start" {
		out = append(out, violation(rec, fmt.Sprintf("stream started with %q, expected message_start", first)))
	}
	if last != "message_stop" {
		out = append(out, violation(rec, fmt.Sprintf("stream ended with %q, expected message_stop", last)))
	}
	openBlocks := map[float64]bool{}
	for _, e := range eventsRaw {
		ev := asMap(e)
		name := asString(ev["event"])
		data := asMap(ev["data"])
		switch name {
		case "content_block_start":
			idx, _ := data["index"].(float64)
			if openBlocks[idx] {
				out = append(out, violation(rec, fmt.Sprintf("content_block_start reopened already-open index %v", idx)))
			}
			openBlocks[idx] = true
		case "content_block_stop":
			idx, _ := data["index"].(float64)
			if !openBlocks[idx] {
				out = append(out, violation(rec, fmt.Sprintf("content_block_stop closed index %v that was never opened", idx)))
			}
			delete(openBlocks, idx)
		}
	}
	for idx := range openBlocks {
		out = append(out, violation(rec, fmt.Sprintf("content_block index %v opened but never closed before message_stop", idx)))
	}
	return out
}
