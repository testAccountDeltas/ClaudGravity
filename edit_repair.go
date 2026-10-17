package main

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// SessionWorkingDirectory tracks the active working directory discovered per session or agent
var (
	sessionCWDLock sync.RWMutex
	sessionCWDMap  = make(map[string]string)
	// sessionCWDSeen bounds sessionCWDMap's otherwise-unbounded growth —
	// each entry is keyed by sessionID:instanceID (one per subagent
	// instance ever seen), and without a sweep this grew forever for the
	// life of the gateway process across every session/subagent that ever
	// connected, same leak class sessionSnapMap already guards against.
	sessionCWDSeen = make(map[string]time.Time)
)

func RecordSessionCWD(sessionID, cwd string) {
	if sessionID == "" || cwd == "" {
		return
	}
	sessionCWDLock.Lock()
	defer sessionCWDLock.Unlock()
	sessionCWDMap[sessionID] = filepath.Clean(strings.TrimSpace(cwd))
	sessionCWDSeen[sessionID] = time.Now()
}

func sweepStaleSessionCWD() {
	sessionCWDLock.Lock()
	defer sessionCWDLock.Unlock()
	cutoff := time.Now().Add(-sessionSnapStaleAfter)
	for sessionID, t := range sessionCWDSeen {
		if t.Before(cutoff) {
			delete(sessionCWDSeen, sessionID)
			delete(sessionCWDMap, sessionID)
		}
	}
}

func GetSessionCWD(sessionID string) string {
	sessionCWDLock.RLock()
	defer sessionCWDLock.RUnlock()
	return sessionCWDMap[sessionID]
}

// isWithinSessionCWD reports whether path is the session's recorded working
// directory or a descendant of it. Returns false (refuse) if the session's
// CWD is unknown, or the file resolution never caught up to produce an
// absolute path — an unconstrained read is worse than a missed repair.
func isWithinSessionCWD(path, sessionID string) bool {
	cwd := GetSessionCWD(sessionID)
	if cwd == "" || !filepath.IsAbs(path) {
		return false
	}
	absCWD, err := filepath.Abs(cwd)
	if err != nil {
		return false
	}
	absPath, err := filepath.Abs(path)
	if err != nil {
		return false
	}
	rel, err := filepath.Rel(absCWD, absPath)
	if err != nil {
		return false
	}
	return rel == "." || (!strings.HasPrefix(rel, "..") && !filepath.IsAbs(rel))
}

// sessionFileSnapshotMap tracks, per session, the last content the MODEL
// ITSELF has seen/produced for a given file path via Read/Write/Edit tool
// calls already present in the conversation history sent with this very
// request. Unlike reading the gateway's own disk, this works identically
// whether the gateway and the client share a filesystem or not -- the
// history is already inside the request body.
const (
	// sessionSnapStaleAfter/sessionSnapSweepEvery bound the otherwise-
	// unbounded growth of sessionSnapMap: without this, every file content
	// a session has ever Read/Written stays in memory for the gateway's
	// entire process lifetime, across every session that ever connected.
	sessionSnapStaleAfter = 2 * time.Hour
	sessionSnapSweepEvery = 30 * time.Minute
)

var (
	sessionSnapLock sync.RWMutex
	sessionSnapMap  = make(map[string]map[string]string) // sessionID -> filePath -> content
	sessionSnapSeen = make(map[string]time.Time)         // sessionID -> last touch, for staleness sweeps
)

func init() {
	go func() {
		ticker := time.NewTicker(sessionSnapSweepEvery)
		defer ticker.Stop()
		for range ticker.C {
			sweepStaleSessionSnapshots()
			sweepStaleSessionCWD()
		}
	}()
}

func sweepStaleSessionSnapshots() {
	sessionSnapLock.Lock()
	defer sessionSnapLock.Unlock()
	cutoff := time.Now().Add(-sessionSnapStaleAfter)
	for sessionID, t := range sessionSnapSeen {
		if t.Before(cutoff) {
			delete(sessionSnapSeen, sessionID)
			delete(sessionSnapMap, sessionID)
		}
	}
}

// canonicalSnapshotKey normalizes a tool-supplied file path before it's used
// as a map key: a model that Read a file as "src/app.ts" and later Edit'd it
// as "src\app.ts" or an absolute "H:\project\src\app.ts" previously missed
// the snapshot entirely on an exact-string-match lookup, since nothing
// normalized slash direction or relative/absolute form between calls.
func canonicalSnapshotKey(path string) string {
	return filepath.Clean(filepath.ToSlash(strings.TrimSpace(path)))
}

func getSessionSnapshot(sessionID, path string) (string, bool) {
	sessionSnapLock.RLock()
	defer sessionSnapLock.RUnlock()
	m := sessionSnapMap[sessionID]
	if m == nil {
		return "", false
	}
	v, ok := m[canonicalSnapshotKey(path)]
	return v, ok
}

func setSessionSnapshot(sessionID, path, content string) {
	if sessionID == "" || path == "" {
		return
	}
	sessionSnapLock.Lock()
	defer sessionSnapLock.Unlock()
	m := sessionSnapMap[sessionID]
	if m == nil {
		m = make(map[string]string)
		sessionSnapMap[sessionID] = m
	}
	m[canonicalSnapshotKey(path)] = content
	sessionSnapSeen[sessionID] = time.Now()
}

// stripCatNPrefix reverses the Read tool's "cat -n"-style numbering
// ("   123\tcontent") back into raw file text. Lines that don't match the
// pattern (stray warnings etc.) are kept as-is rather than dropped.
var catNLinePrefix = regexp.MustCompile(`^\s*\d+\t`)

func stripCatNPrefix(dump string) string {
	lines := strings.Split(dump, "\n")
	out := make([]string, 0, len(lines))
	for _, l := range lines {
		if loc := catNLinePrefix.FindStringIndex(l); loc != nil {
			out = append(out, l[loc[1]:])
		} else {
			out = append(out, l)
		}
	}
	return strings.Join(out, "\n")
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

// RecordSessionFileSnapshots replays the Read/Write/Edit tool calls already
// present in this request's own message history to reconstruct each file's
// last known state AS THE MODEL ITSELF SAW/PRODUCED IT.
//
// This is the actually-correct ground truth for RepairEditToolCall: when
// old_string fails to byte-match, it's because the model mis-transcribed
// whitespace/tabs/newlines from content it already read earlier in this
// same conversation -- not because the file is unreachable or "new". The
// conversation already carries that content over the wire to the gateway
// regardless of where the gateway runs, so this needs no local disk access
// at all (works for a remote-deployed gateway, unlike os.ReadFile below).
func RecordSessionFileSnapshots(sessionID string, bodyBytes []byte) {
	if sessionID == "" {
		return
	}
	messages := gjson.GetBytes(bodyBytes, "messages")
	if !messages.IsArray() {
		return
	}

	toolIDToName := make(map[string]string)
	toolIDToPath := make(map[string]string)

	// A tool_use's own result (is_error:true on a failed old_string match)
	// arrives in a LATER message than the tool_use itself, so it has to be
	// collected in its own pass before the main loop below — which replays
	// Edit's old_string/new_string against the tracked snapshot the moment
	// it sees the tool_use, with no way yet to know whether that edit
	// actually succeeded on the real file. Applying it unconditionally
	// poisoned the snapshot with content that was never actually written,
	// then handed back to later repair attempts as if it were real.
	failedToolIDs := make(map[string]bool)
	messages.ForEach(func(_, msg gjson.Result) bool {
		content := msg.Get("content")
		if !content.IsArray() {
			return true
		}
		content.ForEach(func(_, p gjson.Result) bool {
			if p.Get("type").String() == "tool_result" && p.Get("is_error").Bool() {
				if tid := p.Get("tool_use_id").String(); tid != "" {
					failedToolIDs[tid] = true
				}
			}
			return true
		})
		return true
	})

	messages.ForEach(func(_, msg gjson.Result) bool {
		content := msg.Get("content")
		if !content.IsArray() {
			return true
		}
		content.ForEach(func(_, p gjson.Result) bool {
			switch p.Get("type").String() {
			case "tool_use":
				id := p.Get("id").String()
				name := strings.ToLower(p.Get("name").String())
				if id == "" {
					return true
				}
				toolIDToName[id] = name

				input := p.Get("input")
				path := firstNonEmpty(input.Get("file_path").String(), input.Get("path").String(), input.Get("target_file").String())
				if path == "" {
					return true
				}
				toolIDToPath[id] = path
				if failedToolIDs[id] {
					return true
				}

				switch name {
				case "write":
					// Write hands the full new file content directly in the
					// call itself -- no need to wait for the tool_result.
					if c := input.Get("content"); c.Exists() {
						setSessionSnapshot(sessionID, path, c.String())
					}
				case "edit", "update", "str_replace_editor", "notebookedit":
					// Replay same-turn edits against whatever snapshot we
					// already have, so a second Edit on a file touched
					// earlier in this very request compares against
					// up-to-date content instead of the stale initial Read.
					oldStr := firstNonEmpty(input.Get("old_string").String(), input.Get("old_str").String(), input.Get("pattern").String())
					newStr := firstNonEmpty(input.Get("new_string").String(), input.Get("new_str").String())
					if oldStr != "" {
						if cur, ok := getSessionSnapshot(sessionID, path); ok && strings.Contains(cur, oldStr) {
							setSessionSnapshot(sessionID, path, strings.Replace(cur, oldStr, newStr, 1))
						}
					}
				}

			case "tool_result":
				toolUseID := p.Get("tool_use_id").String()
				if toolIDToName[toolUseID] != "read" {
					return true
				}
				path := toolIDToPath[toolUseID]
				if path == "" {
					return true
				}

				var resText string
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
				if resText != "" {
					setSessionSnapshot(sessionID, path, stripCatNPrefix(resText))
				}
			}
			return true
		})
		return true
	})
}

// ExtractSystemPromptText pulls the plain-text system prompt out of an
// Anthropic request body, decoding whichever shape Claude Code sent it in
// (a bare string, or an array of {"type":"text","text":...} blocks with
// cache_control metadata). Callers must use this — or gjson's .String(), not
// .Raw — before handing text to any regex/string search: .Raw returns the
// literal source JSON bytes with escape sequences un-decoded (a real
// newline inside the prompt is still the two ASCII characters '\' 'n' in
// .Raw, not a newline byte), which silently breaks any "stop at \n" pattern.
func ExtractSystemPromptText(bodyBytes []byte) string {
	sys := gjson.GetBytes(bodyBytes, "system")
	if sys.Type == gjson.String {
		return sys.String()
	}
	if sys.IsArray() {
		var sb strings.Builder
		sys.ForEach(func(_, block gjson.Result) bool {
			sb.WriteString(block.Get("text").String())
			sb.WriteString("\n")
			return true
		})
		return sb.String()
	}
	return ""
}

// ExtractCWDFromSystemPrompt extracts the user session working directory from Claude Code environment preamble
func ExtractCWDFromSystemPrompt(systemPrompt string) string {
	if systemPrompt == "" {
		return ""
	}
	re := regexp.MustCompile(`(?:Primary working directory|Current working directory|working directory)[:\s]+([^\r\n]+)`)
	m := re.FindStringSubmatch(systemPrompt)
	if len(m) > 1 {
		p := strings.TrimSpace(m[1])
		p = strings.Trim(p, "\"'`\\")
		return filepath.Clean(p)
	}
	return ""
}

// normalizeLineForMatch collapses all whitespace (spaces, tabs) into a single space and trims edges
func normalizeLineForMatch(s string) string {
	var b strings.Builder
	inSpace := false
	for _, r := range s {
		if unicode.IsSpace(r) {
			if !inSpace {
				b.WriteByte(' ')
				inSpace = true
			}
		} else {
			b.WriteRune(r)
			inSpace = false
		}
	}
	return strings.TrimSpace(b.String())
}

// splitFileIntoLines splits bytes into lines while preserving their exact line endings
type fileLineInfo struct {
	text   string // original raw line including ending
	clean  string // line without \r and \n
	norm   string // whitespace normalized
	offset int    // byte offset in file
}

// RepairEditToolCall checks if a tool call is an Edit/str_replace_editor/Update call,
// and if old_string does not match byte-for-byte in the file on disk, repairs it using fuzzy whitespace matching.
func RepairEditToolCall(toolName string, argsRaw string, sessionID string) (string, bool) {
	nameLower := strings.ToLower(toolName)
	if nameLower != "edit" && nameLower != "str_replace_editor" && nameLower != "update" && nameLower != "notebookedit" {
		return argsRaw, false
	}

	parsed := gjson.Parse(argsRaw)
	if !parsed.IsObject() {
		return argsRaw, false
	}

	// 1. Locate file path
	targetPath := ""
	for _, k := range []string{"file_path", "path", "notebook_path", "target_file"} {
		if val := parsed.Get(k).String(); val != "" {
			targetPath = val
			break
		}
	}
	if targetPath == "" {
		return argsRaw, false
	}

	// 2. Locate old_string
	oldStrKey := ""
	oldStr := ""
	for _, k := range []string{"old_string", "old_str", "old_text", "pattern"} {
		if val := parsed.Get(k); val.Exists() {
			oldStrKey = k
			oldStr = val.String()
			break
		}
	}
	if oldStrKey == "" || oldStr == "" {
		return argsRaw, false
	}

	// 3. Resolve path against filesystem and session CWD
	resolvedPath := targetPath
	if !filepath.IsAbs(resolvedPath) {
		cwd := GetSessionCWD(sessionID)
		if cwd != "" {
			candidate := filepath.Join(cwd, resolvedPath)
			if _, err := os.Stat(candidate); err == nil {
				resolvedPath = candidate
			}
		}
	}

	// Sandbox: this function reads arbitrary client-supplied paths off the
	// GATEWAY's own disk to compare against old_string. Without a boundary
	// check, any tenant with a valid API key could make the gateway read
	// any file it has OS permission to open (/etc/shadow-equivalent,
	// other tenants' config, ...) just by naming it in a fake Edit call.
	// Require the resolved path to live inside the session's own recorded
	// working directory; if that's unknown, refuse rather than guess.
	//
	// This check guards ONLY the disk-read path below. The snapshot
	// fallback (getSessionSnapshot) touches no disk and exposes nothing the
	// client didn't already send itself in this very request, so it's safe
	// to use even when the sandbox check fails or the gateway has no
	// filesystem access to the client's files at all (remote deployment).
	var fileContent string
	if isWithinSessionCWD(resolvedPath, sessionID) {
		if data, err := os.ReadFile(resolvedPath); err == nil {
			fileContent = string(data)
		}
	}
	if fileContent == "" {
		snap, ok := getSessionSnapshot(sessionID, targetPath)
		if !ok {
			snap, ok = getSessionSnapshot(sessionID, resolvedPath)
		}
		if !ok {
			return argsRaw, false
		}
		fileContent = snap
	}

	// Direct byte-for-byte match already exists! No repair needed.
	if strings.Contains(fileContent, oldStr) {
		return argsRaw, false
	}

	// Locate new_string's key too (same naming pairs as old_string, minus
	// "pattern" which has no replacement-text counterpart), so the line-
	// ending fix below can normalize it along with old_string instead of
	// leaving it as a mismatched style that then gets written into the file.
	newStrKey := ""
	newStr := ""
	for _, k := range []string{"new_string", "new_str", "new_text"} {
		if val := parsed.Get(k); val.Exists() {
			newStrKey = k
			newStr = val.String()
			break
		}
	}

	// 4. Try newline normalization (\r\n vs \n)
	hasCRLF := strings.Contains(fileContent, "\r\n")
	normalizedLF := strings.ReplaceAll(oldStr, "\r\n", "\n")
	if hasCRLF {
		crlfCandidate := strings.ReplaceAll(normalizedLF, "\n", "\r\n")
		if strings.Contains(fileContent, crlfCandidate) {
			repaired, err := sjson.Set(argsRaw, oldStrKey, crlfCandidate)
			if err == nil {
				// Without this, old_string now matches the file's real CRLF
				// line endings but new_string (still LF, as the model wrote
				// it) gets substituted in verbatim, leaving the edited span
				// with LF endings inside an otherwise-CRLF file.
				if newStrKey != "" {
					newCRLFCandidate := strings.ReplaceAll(strings.ReplaceAll(newStr, "\r\n", "\n"), "\n", "\r\n")
					if newCRLFCandidate != newStr {
						if withNew, err2 := sjson.Set(repaired, newStrKey, newCRLFCandidate); err2 == nil {
							repaired = withNew
						}
					}
				}
				LogGateway("[Auto-Repair Edit] Fixed CRLF newlines for %s (len: %d)", filepath.Base(resolvedPath), len(crlfCandidate))
				return repaired, true
			}
		}
	} else {
		if strings.Contains(fileContent, normalizedLF) {
			repaired, err := sjson.Set(argsRaw, oldStrKey, normalizedLF)
			if err == nil {
				if newStrKey != "" {
					normalizedNewLF := strings.ReplaceAll(newStr, "\r\n", "\n")
					if normalizedNewLF != newStr {
						if withNew, err2 := sjson.Set(repaired, newStrKey, normalizedNewLF); err2 == nil {
							repaired = withNew
						}
					}
				}
				LogGateway("[Auto-Repair Edit] Fixed LF newlines for %s (len: %d)", filepath.Base(resolvedPath), len(normalizedLF))
				return repaired, true
			}
		}
	}

	// 5. Advanced Whitespace-Insensitive & Tab Matching (Line-by-line Sliding Window)
	repairedExact, ok := findFuzzyLineMatch(fileContent, oldStr)
	if ok && repairedExact != "" {
		repaired, err := sjson.Set(argsRaw, oldStrKey, repairedExact)
		if err == nil {
			LogGateway("[Auto-Repair Edit] Successfully fuzzy-repaired old_string for %s (%d -> %d bytes, tabs/spaces matched)",
				filepath.Base(resolvedPath), len(oldStr), len(repairedExact))
			return repaired, true
		}
	}

	return argsRaw, false
}

func findFuzzyLineMatch(fileContent, oldStr string) (string, bool) {
	// Parse model lines
	rawOldLines := strings.Split(strings.ReplaceAll(oldStr, "\r\n", "\n"), "\n")
	if len(rawOldLines) == 0 {
		return "", false
	}

	var normOldLines []string
	for _, l := range rawOldLines {
		normOldLines = append(normOldLines, normalizeLineForMatch(l))
	}

	// Parse file lines and track byte offsets
	var fileLines []fileLineInfo
	offset := 0
	rem := fileContent
	for len(rem) > 0 {
		idx := strings.IndexAny(rem, "\r\n")
		var rawLine string
		var cleanLine string
		if idx == -1 {
			rawLine = rem
			cleanLine = rem
			rem = ""
		} else {
			if idx+1 < len(rem) && rem[idx] == '\r' && rem[idx+1] == '\n' {
				rawLine = rem[:idx+2]
				cleanLine = rem[:idx]
				rem = rem[idx+2:]
			} else {
				rawLine = rem[:idx+1]
				cleanLine = rem[:idx]
				rem = rem[idx+1:]
			}
		}
		fileLines = append(fileLines, fileLineInfo{
			text:   rawLine,
			clean:  cleanLine,
			norm:   normalizeLineForMatch(cleanLine),
			offset: offset,
		})
		offset += len(rawLine)
	}

	mLen := len(normOldLines)
	if len(fileLines) < mLen {
		return "", false
	}

	// Sliding window scan
	var matchIndices []int
	for i := 0; i <= len(fileLines)-mLen; i++ {
		match := true
		for j := 0; j < mLen; j++ {
			// If both are empty lines, they match
			if normOldLines[j] == "" && fileLines[i+j].norm == "" {
				continue
			}
			if fileLines[i+j].norm != normOldLines[j] {
				match = false
				break
			}
		}
		if match {
			matchIndices = append(matchIndices, i)
		}
	}

	// Ensure unique, unambiguous match
	if len(matchIndices) == 1 {
		startIdx := matchIndices[0]
		endIdx := startIdx + mLen - 1

		startOffset := fileLines[startIdx].offset
		// normalizeLineForMatch strips leading whitespace for comparison,
		// so a model-written old_string with no leading indentation (e.g.
		// "x = 10;") matches a real file line that DOES have some
		// ("    x = 10;") — but the returned span always started at the raw
		// line's own offset (before that indentation), capturing it as part
		// of old_string. Claude Code then replaces that span with the
		// model's new_string, which is typically written in the same
		// unindented style as its old_string, silently stripping the file's
		// real indentation. If old_string's first line had no leading
		// whitespace of its own, start the returned span after the file
		// line's leading whitespace instead, leaving it as untouched
		// surrounding text outside the replaced span.
		firstOldLine := rawOldLines[0]
		if firstOldLine == strings.TrimLeft(firstOldLine, " \t") {
			firstLineClean := fileLines[startIdx].clean
			leadingLen := len(firstLineClean) - len(strings.TrimLeft(firstLineClean, " \t"))
			startOffset += leadingLen
		}
		lastLine := fileLines[endIdx]

		// If the last line in model's old_string did not end with a newline,
		// strip trailing newline from the matched file segment to match model's boundary intention
		endOffset := lastLine.offset + len(lastLine.text)
		if !strings.HasSuffix(oldStr, "\n") && !strings.HasSuffix(oldStr, "\r") {
			endOffset = lastLine.offset + len(lastLine.clean)
		}

		if endOffset <= len(fileContent) && startOffset <= endOffset {
			return fileContent[startOffset:endOffset], true
		}
	}

	return "", false
}

// RepairGrepToolCall and RepairGlobToolCall catch the specific ways Gemini
// mangles these two tools' calls — confirmed independently in a sibling
// project (opencode, same Gemini models): the model reliably drops or
// misplaces the one required "pattern" field on a multi-field search tool,
// while a single general shell command string never has this problem. Not
// wired to any live behavior change by itself — strip_search_tools stays
// true by default (see README.md) until there's been real traffic through
// this repair pass to build confidence in it.

// RepairGrepToolCall fixes: (1) the model passing a bare JSON string instead
// of an object — wrapped as {"pattern": <string>}; (2) a missing/empty
// "pattern" with the search term sitting in "path" instead (path containing
// glob/regex metacharacters, or no path separator at all, is the tell).
func RepairGrepToolCall(argsRaw string) (string, bool) {
	parsed := gjson.Parse(argsRaw)
	if !parsed.Exists() {
		return argsRaw, false
	}
	if parsed.Type == gjson.String {
		s := parsed.String()
		if s == "" {
			return argsRaw, false
		}
		fixed, _ := sjson.Set(`{}`, "pattern", s)
		return fixed, true
	}
	if !parsed.IsObject() {
		return argsRaw, false
	}

	pattern := parsed.Get("pattern").String()
	path := parsed.Get("path").String()
	if pattern != "" {
		return argsRaw, false
	}
	if path != "" && (strings.ContainsAny(path, "*?[]()|^$+\\") || !strings.ContainsAny(path, "/\\")) {
		fixed, _ := sjson.Set(argsRaw, "pattern", path)
		fixed, _ = sjson.Delete(fixed, "path")
		return fixed, true
	}
	return argsRaw, false
}

// RepairGlobToolCall fixes the same missing/misplaced "pattern" issue for
// Glob — its search term is a filename glob (e.g. "**/*.go") rather than a
// regex, so the "looks like a pattern" heuristic checks glob wildcards.
func RepairGlobToolCall(argsRaw string) (string, bool) {
	parsed := gjson.Parse(argsRaw)
	if !parsed.Exists() {
		return argsRaw, false
	}
	if parsed.Type == gjson.String {
		s := parsed.String()
		if s == "" {
			return argsRaw, false
		}
		fixed, _ := sjson.Set(`{}`, "pattern", s)
		return fixed, true
	}
	if !parsed.IsObject() {
		return argsRaw, false
	}

	pattern := parsed.Get("pattern").String()
	path := parsed.Get("path").String()
	if pattern != "" {
		return argsRaw, false
	}
	if path != "" && strings.ContainsAny(path, "*?[]") {
		fixed, _ := sjson.Set(argsRaw, "pattern", path)
		fixed, _ = sjson.Delete(fixed, "path")
		return fixed, true
	}
	return argsRaw, false
}
