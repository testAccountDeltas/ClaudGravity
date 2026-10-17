// speccheck statically checks real gateway traffic (captured by
// speccapture.go into request_captures/*.ndjson) against a growing list
// of specification invariants — see AGENT_HANDOFF_NOTES.md for why this
// exists. It reads the NDJSON format generically (no import of the
// gateway's own "main" package, by design — this stays a decoupled,
// read-only tool) and never needs network access or live credentials.
//
// Usage:
//
//	go run ./cmd/speccheck -dir request_captures
//	go run ./cmd/speccheck -dir request_captures -since 2026-10-18T00:00:00Z -rules RoleAlternation,ToolCallResponseParity
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// CaptureRecord is one line of request_captures/*.ndjson, decoded
// generically — it never needs to know the gateway's own Go types, only
// the JSON shape speccapture.go writes.
type CaptureRecord map[string]interface{}

func (r CaptureRecord) rid() string {
	if v, ok := r["rid"]; ok {
		return fmt.Sprint(v)
	}
	return "?"
}

func (r CaptureRecord) timestamp() string {
	if v, ok := r["timestamp"].(string); ok {
		return v
	}
	return ""
}

// snapshotBody returns the "body" field of a top-level leg ("claude_request",
// "gemini_request", "gemini_response", "claude_response"), or nil if that
// leg wasn't captured for this request.
func (r CaptureRecord) snapshotBody(leg string) interface{} {
	leg2, ok := r[leg].(map[string]interface{})
	if !ok {
		return nil
	}
	return leg2["body"]
}

func (r CaptureRecord) snapshotHeaders(leg string) map[string]interface{} {
	leg2, ok := r[leg].(map[string]interface{})
	if !ok {
		return nil
	}
	h, _ := leg2["headers"].(map[string]interface{})
	return h
}

type Violation struct {
	Rule      string
	RID       string
	Timestamp string
	Source    string // which .ndjson file this came from
	Detail    string
}

func main() {
	dir := flag.String("dir", "request_captures", "directory of *.ndjson capture files")
	since := flag.String("since", "", "RFC3339 timestamp; skip records before this")
	until := flag.String("until", "", "RFC3339 timestamp; skip records at/after this")
	rulesFlag := flag.String("rules", "all", "comma-separated rule names to run, or \"all\"")
	listRules := flag.Bool("list-rules", false, "print the available rule names and exit")
	flag.Parse()

	if *listRules {
		for _, r := range AllRules {
			fmt.Println(r.Name)
		}
		return
	}

	selected := AllRules
	if *rulesFlag != "all" && *rulesFlag != "" {
		want := make(map[string]bool)
		for _, n := range strings.Split(*rulesFlag, ",") {
			want[strings.TrimSpace(n)] = true
		}
		selected = nil
		for _, r := range AllRules {
			if want[r.Name] {
				selected = append(selected, r)
			}
		}
	}

	var sinceT, untilT time.Time
	var err error
	if *since != "" {
		sinceT, err = time.Parse(time.RFC3339, *since)
		if err != nil {
			fmt.Fprintf(os.Stderr, "bad -since: %v\n", err)
			os.Exit(2)
		}
	}
	if *until != "" {
		untilT, err = time.Parse(time.RFC3339, *until)
		if err != nil {
			fmt.Fprintf(os.Stderr, "bad -until: %v\n", err)
			os.Exit(2)
		}
	}

	files, err := filepath.Glob(filepath.Join(*dir, "*.ndjson"))
	if err != nil || len(files) == 0 {
		fmt.Fprintf(os.Stderr, "no *.ndjson files found in %s\n", *dir)
		os.Exit(1)
	}
	sort.Strings(files)

	var allViolations []Violation
	recordCount := 0
	for _, path := range files {
		f, err := os.Open(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "skipping %s: %v\n", path, err)
			continue
		}
		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 1024*1024), 64*1024*1024)
		for scanner.Scan() {
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}
			var rec CaptureRecord
			if err := json.Unmarshal(line, &rec); err != nil {
				fmt.Fprintf(os.Stderr, "%s: malformed line skipped: %v\n", path, err)
				continue
			}
			if !sinceT.IsZero() || !untilT.IsZero() {
				ts, err := time.Parse(time.RFC3339Nano, rec.timestamp())
				if err == nil {
					if !sinceT.IsZero() && ts.Before(sinceT) {
						continue
					}
					if !untilT.IsZero() && !ts.Before(untilT) {
						continue
					}
				}
			}
			recordCount++
			for _, rule := range selected {
				for _, v := range rule.Check(rec) {
					v.Rule = rule.Name
					v.Source = filepath.Base(path)
					allViolations = append(allViolations, v)
				}
			}
		}
		_ = f.Close()
	}

	fmt.Printf("speccheck: %d record(s) across %d file(s), %d rule(s) run\n", recordCount, len(files), len(selected))
	if len(allViolations) == 0 {
		fmt.Println("no violations found")
		return
	}

	byRule := make(map[string][]Violation)
	for _, v := range allViolations {
		byRule[v.Rule] = append(byRule[v.Rule], v)
	}
	ruleNames := make([]string, 0, len(byRule))
	for name := range byRule {
		ruleNames = append(ruleNames, name)
	}
	sort.Strings(ruleNames)

	for _, name := range ruleNames {
		vs := byRule[name]
		fmt.Printf("\n=== %s (%d violation(s)) ===\n", name, len(vs))
		for _, v := range vs {
			fmt.Printf("  [%s rid=%s %s] %s\n", v.Source, v.RID, v.Timestamp, v.Detail)
		}
	}
	fmt.Printf("\n%d total violation(s)\n", len(allViolations))
	os.Exit(1)
}
