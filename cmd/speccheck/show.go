package main

import (
	"encoding/json"
	"fmt"
	"strings"
)

// showRecord prints one captured request as a readable, chronological
// walkthrough of all four legs — this is the "what came in, what went
// out, what the gateway did to it, what came back, what the client
// actually got" view the automated rule-checking report doesn't give you.
// Bodies are pretty-printed; streaming legs (chunks/events) are collapsed
// into a compact sequence summary instead of dumped raw, since a real
// streamed response can be hundreds of SSE events and the point here is
// readability, not completeness (the raw NDJSON line is still right there
// in the capture file if that's ever needed).
func showRecord(rec CaptureRecord) {
	fmt.Println(strings.Repeat("=", 78))
	header := fmt.Sprintf("rid=%s  %s", rec.rid(), rec.timestamp())
	if sid, ok := rec["session_id"].(string); ok && sid != "" {
		header += "  session=" + sid
	}
	if ak, ok := rec["agent_key"].(string); ok && ak != "" {
		header += "  agent=" + ak
	}
	fmt.Println(header)
	fmt.Println(strings.Repeat("=", 78))

	showLeg(rec, "claude_request", "1. Claude Code -> Gateway (incoming request)")
	showLeg(rec, "gemini_request", "2. Gateway -> Gemini (translated request)")
	showResponseLeg(rec, "gemini_response", "3. Gemini -> Gateway (upstream response)")
	showResponseLeg(rec, "claude_response", "4. Gateway -> Claude Code (final response)")
	fmt.Println()
}

func showLeg(rec CaptureRecord, key, title string) {
	leg, ok := rec[key].(map[string]interface{})
	if !ok {
		fmt.Printf("\n--- %s: not captured ---\n", title)
		return
	}
	fmt.Printf("\n--- %s ---\n", title)
	if headers, ok := leg["headers"].(map[string]interface{}); ok && len(headers) > 0 {
		fmt.Println("headers:")
		printIndentedJSON(headers, "  ")
	}
	if body, ok := leg["body"]; ok {
		fmt.Println("body:")
		printIndentedJSON(body, "  ")
	}
}

// showResponseLeg handles the two legs that can ALSO be a streamed
// accumulation ({"chunks":[...]} / {"events":[...]}) instead of one body.
func showResponseLeg(rec CaptureRecord, key, title string) {
	legRaw, ok := rec[key]
	if !ok {
		fmt.Printf("\n--- %s: not captured ---\n", title)
		return
	}
	leg, ok := legRaw.(map[string]interface{})
	if !ok {
		fmt.Printf("\n--- %s: not captured ---\n", title)
		return
	}
	fmt.Printf("\n--- %s ---\n", title)
	if headers, ok := leg["headers"].(map[string]interface{}); ok && len(headers) > 0 {
		fmt.Println("headers:")
		printIndentedJSON(headers, "  ")
	}
	if body, ok := leg["body"]; ok {
		fmt.Println("body:")
		printIndentedJSON(body, "  ")
		return
	}
	if chunks, ok := leg["chunks"].([]interface{}); ok {
		fmt.Printf("streamed: %d Gemini chunk(s)\n", len(chunks))
		for i, c := range chunks {
			fmt.Printf("  chunk[%d]:\n", i)
			printIndentedJSON(c, "    ")
		}
		return
	}
	if events, ok := leg["events"].([]interface{}); ok {
		fmt.Printf("streamed: %d SSE event(s):\n", len(events))
		printEventSequence(events)
	}
}

// printEventSequence collapses consecutive same-named events (overwhelmingly
// content_block_delta in any real response) into one "x N" line, and shows
// every other event's structural fields (index, stop_reason, usage, ...) in
// full — the deltas' actual text is already stripped by redactSkeleton, so
// showing each one individually would just be N near-identical noise lines.
func printEventSequence(events []interface{}) {
	i := 0
	for i < len(events) {
		ev := asMap(events[i])
		name := asString(ev["event"])
		j := i
		for j < len(events) && asString(asMap(events[j])["event"]) == name {
			j++
		}
		count := j - i
		if count > 1 {
			fmt.Printf("  %s  x%d\n", name, count)
		} else {
			fmt.Printf("  %s\n", name)
			if data, ok := ev["data"]; ok {
				printIndentedJSON(data, "    ")
			}
		}
		i = j
	}
}

func printIndentedJSON(v interface{}, indent string) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		fmt.Printf("%s<unprintable: %v>\n", indent, err)
		return
	}
	for _, line := range strings.Split(string(b), "\n") {
		fmt.Println(indent + line)
	}
}
