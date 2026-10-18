// normalize reads real captured traffic from request_captures/*.ndjson
// (written by speccapture.go — already redacted of generative text, but
// NOT yet stripped of fields that are legitimately different every single
// time the same code runs against the same input: the message id the
// streamer generates fresh per response, tool-call ids minted per call,
// the gateway's own per-request timestamp, and Gemini's own thought
// signature) and produces the Phase 0 characterization corpus described in
// ARCHITECTURE_V2_SPECIFICATION_AND_MIGRATION_PLAN.md: a selection of real
// sessions, masked so that diffing v1's output against a future v2's
// output on the same frozen input doesn't flag these fields as
// regressions just because they were, by construction, never going to
// match byte-for-byte.
//
// This is a one-shot snapshot tool, not a long-running service: run it
// once now to freeze today's v1 behavior as the golden baseline, before a
// single line of ClaudGravityNext exists.
//
// Usage:
//
//	go run ./cmd/normalize -dir request_captures -out testdata/characterization -count 50
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// idPattern matches Anthropic-style generated ids (msg_..., toolu_...) —
// the two prefixes named explicitly in the spec's Phase 0 section. Masked
// by prefix (not deleted) so a snapshot reader can still tell "this is a
// message id" from "this is a tool-call id" without the exact value
// mattering.
var idPattern = regexp.MustCompile(`^(msg|toolu)_[A-Za-z0-9_-]+$`)

// maskedKeys are JSON object keys whose VALUE is masked unconditionally
// wherever they appear, regardless of nesting — these are exactly the
// four non-deterministic field classes named in Part 7.1 of the spec.
func isMaskedKey(key string) bool {
	lk := strings.ToLower(key)
	if lk == "timestamp" || lk == "thoughtsignature" || lk == "thought_signature" {
		return true
	}
	// Header map keys are stored as-written by net/http's canonicalized
	// form (e.g. "Request-Id"), so match case-insensitively on content
	// rather than on one exact casing.
	return strings.Contains(lk, "request-id") || strings.Contains(lk, "requestid")
}

// normalize recursively walks a decoded JSON value (the same
// map[string]interface{}/[]interface{} shape speccapture.go already
// writes) and masks the non-deterministic fields described above,
// leaving every other field — structure, enums, counts, short strings —
// untouched. This is deliberately the same "redact by rule, not by a
// hardcoded per-field allowlist" philosophy speccapture.go's own
// redactSkeleton already uses, applied to a different field class.
func normalize(v interface{}) interface{} {
	switch t := v.(type) {
	case map[string]interface{}:
		// Generated ids can appear as a MAP KEY, not just a value — e.g.
		// Claude Code's own "tool_uses" status block is keyed by
		// toolu_* id directly (`"tool_uses":{"toolu_...":{"outcome":...}}}`).
		// Keys are masked too, numbered deterministically within this one
		// object (sorted key order) so two masked keys in the same map
		// don't collide into one lost entry.
		keys := make([]string, 0, len(t))
		for k := range t {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		out := make(map[string]interface{}, len(t))
		idKeySeen := map[string]int{}
		for _, k := range keys {
			val := t[k]
			newKey := k
			if idPattern.MatchString(k) {
				prefix := strings.SplitN(k, "_", 2)[0]
				idKeySeen[prefix]++
				newKey = fmt.Sprintf("%s_NORMALIZED_%d", prefix, idKeySeen[prefix])
			}
			if isMaskedKey(k) {
				if s, ok := val.(string); ok && s != "" {
					out[newKey] = fmt.Sprintf("<normalized:%s>", strings.ToLower(k))
					continue
				}
			}
			out[newKey] = normalize(val)
		}
		return out
	case []interface{}:
		out := make([]interface{}, len(t))
		for i, val := range t {
			out[i] = normalize(val)
		}
		return out
	case string:
		if idPattern.MatchString(t) {
			prefix := strings.SplitN(t, "_", 2)[0]
			return prefix + "_NORMALIZED"
		}
		return t
	default:
		return v
	}
}

type record struct {
	raw    map[string]interface{}
	source string
	rid    string
}

func loadAllRecords(dir string) ([]record, error) {
	files, err := filepath.Glob(filepath.Join(dir, "*.ndjson"))
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	var out []record
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
			var m map[string]interface{}
			if err := json.Unmarshal(line, &m); err != nil {
				continue
			}
			out = append(out, record{raw: m, source: filepath.Base(path), rid: fmt.Sprint(m["rid"])})
		}
		_ = f.Close()
	}
	return out, nil
}

// tagsFor classifies a record by a handful of shape markers that matter
// for replay/characterization coverage: streaming vs non-streaming, tool
// use/results present, and multimodal (image/document) attachments —
// exactly the request shapes that have broken repeatedly across 15 audit
// rounds (see the spec's Part 1 diagnosis), so the frozen corpus should
// not be 50 copies of the same plain-text request.
func tagsFor(blob string) map[string]bool {
	tags := map[string]bool{}
	if strings.Contains(blob, `"chunks"`) || strings.Contains(blob, `"events"`) {
		tags["streaming"] = true
	} else {
		tags["nonstreaming"] = true
	}
	if strings.Contains(blob, `"tool_use"`) {
		tags["tool_use"] = true
	}
	if strings.Contains(blob, `"tool_result"`) {
		tags["tool_result"] = true
	}
	if strings.Contains(blob, `"type":"image"`) || strings.Contains(blob, `"media_type":"image`) {
		tags["image"] = true
	}
	if strings.Contains(blob, `"media_type":"application/pdf"`) {
		tags["pdf"] = true
	}
	if strings.Contains(blob, `"thinking"`) || strings.Contains(blob, `thoughtSignature`) {
		tags["thinking"] = true
	}
	return tags
}

func main() {
	dir := flag.String("dir", "request_captures", "directory of *.ndjson capture files to read")
	out := flag.String("out", "testdata/characterization", "output directory for the golden snapshot corpus")
	count := flag.Int("count", 50, "number of sessions to select")
	flag.Parse()

	records, err := loadAllRecords(*dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if len(records) == 0 {
		fmt.Fprintf(os.Stderr, "no records found in %s\n", *dir)
		os.Exit(1)
	}

	// Normalize every candidate once up front, then tag the normalized
	// form (masking must happen before tagging so a masked
	// thoughtSignature value doesn't accidentally defeat its own
	// "thinking" tag).
	type candidate struct {
		rec  record
		norm map[string]interface{}
		blob string
		tags map[string]bool
	}
	cands := make([]candidate, 0, len(records))
	for _, r := range records {
		norm, ok := normalize(r.raw).(map[string]interface{})
		if !ok {
			continue
		}
		b, _ := json.Marshal(norm)
		blob := string(b)
		cands = append(cands, candidate{rec: r, norm: norm, blob: blob, tags: tagsFor(blob)})
	}

	// Greedy diverse selection: first pass picks one record per
	// not-yet-covered tag (in first-seen order), second pass fills the
	// rest with whatever's left, most recent file first (files are
	// already date-named and sorted, so later index == more recent).
	selected := make([]candidate, 0, *count)
	picked := make(map[int]bool)
	covered := map[string]bool{}
	for i, c := range cands {
		if len(selected) >= *count {
			break
		}
		newTag := false
		for t := range c.tags {
			if !covered[t] {
				newTag = true
				break
			}
		}
		if newTag {
			selected = append(selected, c)
			picked[i] = true
			for t := range c.tags {
				covered[t] = true
			}
		}
	}
	for i := len(cands) - 1; i >= 0 && len(selected) < *count; i-- {
		if !picked[i] {
			selected = append(selected, cands[i])
			picked[i] = true
		}
	}

	if err := os.MkdirAll(*out, 0755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	type indexEntry struct {
		File   string   `json:"file"`
		Source string   `json:"source_ndjson"`
		RID    string   `json:"rid"`
		Tags   []string `json:"tags"`
	}
	var index []indexEntry
	for i, c := range selected {
		fname := fmt.Sprintf("%03d_rid%s.json", i+1, c.rec.rid)
		path := filepath.Join(*out, fname)
		data, err := json.MarshalIndent(c.norm, "", "  ")
		if err != nil {
			fmt.Fprintf(os.Stderr, "skip rid=%s: %v\n", c.rec.rid, err)
			continue
		}
		if err := os.WriteFile(path, data, 0644); err != nil {
			fmt.Fprintf(os.Stderr, "write %s: %v\n", path, err)
			continue
		}
		tagNames := make([]string, 0, len(c.tags))
		for t := range c.tags {
			tagNames = append(tagNames, t)
		}
		sort.Strings(tagNames)
		index = append(index, indexEntry{File: fname, Source: c.rec.source, RID: c.rec.rid, Tags: tagNames})
	}
	indexData, _ := json.MarshalIndent(index, "", "  ")
	if err := os.WriteFile(filepath.Join(*out, "index.json"), indexData, 0644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	tagCounts := map[string]int{}
	for _, c := range selected {
		for t := range c.tags {
			tagCounts[t]++
		}
	}
	fmt.Printf("normalize: wrote %d session(s) to %s (from %d candidate records across request_captures)\n", len(selected), *out, len(cands))
	fmt.Println("tag coverage in selection:")
	tagNames := make([]string, 0, len(tagCounts))
	for t := range tagCounts {
		tagNames = append(tagNames, t)
	}
	sort.Strings(tagNames)
	for _, t := range tagNames {
		fmt.Printf("  %-12s %d\n", t, tagCounts[t])
	}
}
