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

// parentSessionKey strips a subagent's ":instanceID" suffix, returning the
// bare parent session key it was derived from (see server.go's cwdKey/
// fileSnapKey construction: "sessionID" for the main thread, "sessionID:
// instanceID" for a subagent instance). Returns ("", false) for a key with
// no such suffix (nothing to fall back to).
func parentSessionKey(sessionID string) (string, bool) {
	idx := strings.LastIndex(sessionID, ":")
	if idx < 0 {
		return "", false
	}
	return sessionID[:idx], true
}

func GetSessionCWD(sessionID string) string {
	sessionCWDLock.RLock()
	defer sessionCWDLock.RUnlock()
	if cwd := sessionCWDMap[sessionID]; cwd != "" {
		return cwd
	}
	// An audit caught that a subagent's OWN first-ever edit call, before
	// it has recorded any CWD of its own, had no fallback at all — the
	// main thread's (or the subagent's own earlier-created parent
	// session's) already-known CWD, recorded under the bare key this
	// subagent's own composite key was literally derived FROM, went
	// completely unused. Safe to fall back to: the parent/child
	// relationship is established by this key's own naming convention,
	// not a cross-tenant guess.
	if parent, ok := parentSessionKey(sessionID); ok {
		return sessionCWDMap[parent]
	}
	return ""
}

func isAbsPath(p string) bool {
	if filepath.IsAbs(p) {
		return true
	}
	if len(p) >= 3 && ((p[0] >= 'a' && p[0] <= 'z') || (p[0] >= 'A' && p[0] <= 'Z')) && p[1] == ':' && (p[2] == '\\' || p[2] == '/') {
		return true
	}
	if len(p) >= 2 && (p[0] == '\\' || p[0] == '/') && p[0] == p[1] {
		return true
	}
	return false
}

func joinCWD(cwd, path string) string {
	if isAbsPath(path) {
		return path
	}
	if strings.Contains(cwd, "\\") {
		return strings.TrimRight(cwd, "\\/") + "\\" + strings.TrimLeft(path, "\\/")
	}
	return filepath.Join(cwd, path)
}

// isWithinSessionCWD reports whether path is the session's recorded working
// directory or a descendant of it. Returns false (refuse) if the session's
// CWD is unknown, or the file resolution never caught up to produce an
// absolute path — an unconstrained read is worse than a missed repair.
func isWithinSessionCWD(path, sessionID string) bool {
	cwd := GetSessionCWD(sessionID)
	if cwd == "" || !isAbsPath(path) {
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
	// sessionSnapFull tracks whether the stored content is a COMPLETE file
	// (a Write, or a Read with no offset/limit) versus a partial slice (a
	// Read with offset/limit set). An audit caught that the original code
	// only ever checked "does ANY snapshot exist", so once the file's
	// FIRST read happened to be partial, every later read — including a
	// different partial slice or even a later full read's own repeat —
	// was silently discarded as "already have a snapshot", permanently
	// freezing the fuzzy-match baseline to whatever that first slice
	// contained. A later full read now always overwrites; a later partial
	// read only overwrites when no full snapshot already won. This does
	// NOT merge two different partial ranges of the same file into one —
	// a second partial read of a different range still replaces rather
	// than extends the first — that's a known remaining limitation, not
	// claimed fixed here.
	sessionSnapFull = make(map[string]map[string]bool)
	sessionSnapSeen = make(map[string]time.Time) // sessionID -> last touch, for staleness sweeps
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
			delete(sessionSnapFull, sessionID)
		}
	}
}

// canonicalSnapshotKey normalizes a tool-supplied file path before it's used
// as a map key: a model that Read a file as "src/app.ts" and later Edit'd it
// as "src\app.ts" or an absolute "H:\project\src\app.ts" previously missed
// the snapshot entirely on an exact-string-match lookup, since nothing
// normalized slash direction or relative/absolute form between calls.
func canonicalSnapshotKey(path string) string {
	// An audit caught that filepath.ToSlash is a no-op on this binary's
	// OWN build target, not the path's origin: filepath.Separator is '/'
	// on a GOOS=linux build (the VPS instance), so ToSlash returns its
	// input completely unchanged regardless of whether the path itself
	// still has Windows backslashes — a path recorded from a Windows
	// client's own absolute path (or just read with backslashes in the
	// session) never got normalized on the Linux deployment. The
	// snapshot was stored under "src\\utils\\parser.go" while lookups
	// searched for "src/utils/parser.go", so every snapshot silently
	// stopped being found on the remote instance. Strip backslashes
	// explicitly, independent of which OS this binary itself targets.
	p := strings.ReplaceAll(strings.TrimSpace(path), "\\", "/")
	// A follow-up audit caught the Windows drive letter itself wasn't
	// normalized either: "H:\..." and "h:\..." are the same real path on
	// Windows (drive letters are case-insensitive) but produced two
	// different cache keys — a snapshot recorded via one casing silently
	// stopped being found by a lookup using the other. Only the drive
	// letter gets lowercased; the rest of the path is left alone since
	// Linux/macOS paths genuinely are case-sensitive.
	if len(p) >= 2 && p[1] == ':' && p[0] >= 'A' && p[0] <= 'Z' {
		p = strings.ToLower(p[:1]) + p[1:]
	}
	return filepath.Clean(p)
}

func getSessionSnapshot(sessionID, path string) (string, bool) {
	sessionSnapLock.RLock()
	defer sessionSnapLock.RUnlock()
	key := canonicalSnapshotKey(path)
	if m := sessionSnapMap[sessionID]; m != nil {
		if v, ok := m[key]; ok {
			return v, true
		}
	}
	// Same parent-session fallback as GetSessionCWD, for the same reason:
	// a subagent's own first edit call has no snapshots of its own yet,
	// but the parent session it was spawned from (and read the same file
	// in) already does.
	if parent, ok := parentSessionKey(sessionID); ok {
		if m := sessionSnapMap[parent]; m != nil {
			v, ok := m[key]
			return v, ok
		}
	}
	return "", false
}

// isSessionSnapshotFull reports whether the currently stored snapshot for
// path is a complete file (see sessionSnapFull's doc comment). False for
// both "no snapshot at all" and "a partial one" — callers that need to
// distinguish those two already have getSessionSnapshot's own ok bool.
func isSessionSnapshotFull(sessionID, path string) bool {
	sessionSnapLock.RLock()
	defer sessionSnapLock.RUnlock()
	m := sessionSnapFull[sessionID]
	if m == nil {
		return false
	}
	return m[canonicalSnapshotKey(path)]
}

func setSessionSnapshot(sessionID, path, content string, isFull bool) {
	if sessionID == "" || path == "" {
		return
	}
	sessionSnapLock.Lock()
	defer sessionSnapLock.Unlock()
	key := canonicalSnapshotKey(path)
	m := sessionSnapMap[sessionID]
	if m == nil {
		m = make(map[string]string)
		sessionSnapMap[sessionID] = m
	}
	m[key] = content
	fm := sessionSnapFull[sessionID]
	if fm == nil {
		fm = make(map[string]bool)
		sessionSnapFull[sessionID] = fm
	}
	fm[key] = isFull
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
	toolIDIsPartial := make(map[string]bool)

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
				// notebook_path added — an audit caught that RepairEditToolCall
				// (the function that actually USES these snapshots) already
				// recognized it, but this recording pass didn't, so a
				// NotebookEdit tool call's file was never tracked at all.
				path := firstNonEmpty(input.Get("file_path").String(), input.Get("path").String(), input.Get("target_file").String(), input.Get("notebook_path").String())
				if path == "" {
					return true
				}
				toolIDToPath[id] = path
				if name == "read" && (input.Get("offset").Exists() || input.Get("limit").Exists()) {
					toolIDIsPartial[id] = true
				}
				if failedToolIDs[id] {
					return true
				}

				switch name {
				case "write":
					// Write hands the full new file content directly in the
					// call itself -- no need to wait for the tool_result.
					if c := input.Get("content"); c.Exists() {
						contentStr := c.String()
						setSessionSnapshot(sessionID, path, contentStr, true)
						if !isAbsPath(path) {
							if cwd := GetSessionCWD(sessionID); cwd != "" {
								setSessionSnapshot(sessionID, joinCWD(cwd, path), contentStr, true)
							}
						}
					}
				case "edit", "update", "str_replace_editor", "notebookedit":
					// Replay same-turn edits against whatever snapshot we
					// already have, so a second Edit on a file touched
					// earlier in this very request compares against
					// up-to-date content instead of the stale initial Read.
					// old_text/new_text added — same gap as notebook_path
					// above: RepairEditToolCall already recognized these
					// (NotebookEdit's own param names), this recording pass
					// didn't.
					oldStr := firstNonEmpty(input.Get("old_string").String(), input.Get("old_str").String(), input.Get("pattern").String(), input.Get("old_text").String())
					newStr := firstNonEmpty(input.Get("new_string").String(), input.Get("new_str").String(), input.Get("new_text").String())
					if oldStr != "" {
						// An audit caught this lookup only ever tried the
						// literal `path` as given — if an earlier Read in
						// this same session recorded its snapshot under an
						// ABSOLUTE path (Claude Code's own tools normally
						// use absolute paths) and this same-turn replay then
						// names the file with a RELATIVE path instead, the
						// lookup missed the existing snapshot entirely,
						// even though the write-back two lines below this
						// already resolves against session CWD. Try the
						// CWD-resolved candidate too, same as
						// RepairEditToolCall's own snapshot lookup does.
						lookupPath := path
						cur, ok := getSessionSnapshot(sessionID, lookupPath)
						if !ok && !isAbsPath(path) {
							if cwd := GetSessionCWD(sessionID); cwd != "" {
								if resolved, rok := getSessionSnapshot(sessionID, joinCWD(cwd, path)); rok {
									cur, ok, lookupPath = resolved, true, joinCWD(cwd, path)
								}
							}
						}
						// An audit caught this compared cur/oldStr with no
						// \r\n vs \n normalization — a snapshot stored with
						// LF (the common case) never matched an incoming
						// old_string using CRLF (a Windows client), or vice
						// versa, even when the actual text content was
						// identical, silently skipping the snapshot update
						// and leaving later same-turn edits comparing
						// against stale content.
						matchOldStr, matchNewStr := oldStr, newStr
						matchCur := cur
						if ok && !strings.Contains(matchCur, matchOldStr) {
							normCur := strings.ReplaceAll(matchCur, "\r\n", "\n")
							normOld := strings.ReplaceAll(matchOldStr, "\r\n", "\n")
							if strings.Contains(normCur, normOld) {
								matchCur = normCur
								matchOldStr = normOld
								matchNewStr = strings.ReplaceAll(matchNewStr, "\r\n", "\n")
							}
						}
						if ok && strings.Contains(matchCur, matchOldStr) {
							n := 1
							if input.Get("replace_all").Bool() {
								n = -1
							}
							updated := strings.Replace(matchCur, matchOldStr, matchNewStr, n)
							// Preserves whatever fullness the snapshot being
							// replayed-against already had — editing text
							// within a partial slice doesn't make it complete.
							wasFull := isSessionSnapshotFull(sessionID, lookupPath)
							setSessionSnapshot(sessionID, path, updated, wasFull)
							if !isAbsPath(path) {
								if cwd := GetSessionCWD(sessionID); cwd != "" {
									setSessionSnapshot(sessionID, joinCWD(cwd, path), updated, wasFull)
								}
							}
						}
					}
				}

			case "tool_result":
				// Do not record snapshot if the tool call failed (e.g. file not found)
				if p.Get("is_error").Bool() || failedToolIDs[p.Get("tool_use_id").String()] {
					return true
				}
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
					cleanContent := stripCatNPrefix(resText)
					isPartial := toolIDIsPartial[toolUseID]
					// An audit caught that a Read with no explicit
					// offset/limit still silently gets truncated at 2000
					// lines by Claude Code's own tool — toolIDIsPartial
					// only tracks whether OFFSET/LIMIT was given, not
					// whether the RESULT was actually truncated, so this
					// case was marked isFull=true (via !isPartial below)
					// even though lines 2001+ don't exist here at all. A
					// later explicit read of those lines then got silently
					// discarded by the isSessionSnapshotFull check, since
					// the snapshot already claimed to be complete. This
					// doesn't fully solve 2000+-line files (still needs
					// real line-indexed snapshot storage to merge ranges
					// instead of replacing them — a larger, deliberately
					// deferred restructuring) but it at least lets a
					// later explicit read of the missing tail actually
					// register instead of being guaranteed-discarded.
					if !isPartial && strings.Count(cleanContent, "\n") >= 2000 {
						isPartial = true
					}
					// Was: `_, hasFullSnap := getSessionSnapshot(...)` — that
					// only ever reported "does ANY snapshot exist", full or
					// partial, so once this file's first Read happened to be
					// partial (now the common case for any file long enough
					// to need the 500+-line chunks readToolGuidance pushes
					// toward), every later Read of it — including a
					// DIFFERENT partial slice — was discarded as "already
					// have one", permanently freezing the fuzzy-match
					// baseline to that first slice. A full read always
					// wins; a partial one only wins when no full snapshot
					// already exists (still doesn't merge two different
					// partial ranges together — see sessionSnapFull's doc
					// comment).
					if !isPartial || !isSessionSnapshotFull(sessionID, path) {
						setSessionSnapshot(sessionID, path, cleanContent, !isPartial)
						if !isAbsPath(path) {
							if cwd := GetSessionCWD(sessionID); cwd != "" {
								setSessionSnapshot(sessionID, joinCWD(cwd, path), cleanContent, !isPartial)
							}
						}
					}
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

// normalizeLineForMatch collapses all whitespace (spaces, tabs) into a single space,
// normalizes typographical/curly quotes and dashes, and trims edges.
func normalizeLineForMatch(s string) string {
	var b strings.Builder
	inSpace := false
	for _, r := range s {
		switch r {
		case '“', '”', '«', '»': // Left/right double curly quotes, guillemets
			r = '"'
		case '‘', '’', '‚', '‛': // Left/right single curly quotes
			r = '\''
		case '–', '—': // En-dash, em-dash
			r = '-'
		case ' ': // Non-breaking space
			r = ' '
		}
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
// RepairEditToolCall repairs a fuzzy-matched old_string/new_string against
// the real file, then normalizes whatever alias parameter names the model
// used (old_str/new_str/path/notebook_path/target_file/pattern/old_text/
// new_text) to the exact names Claude Code's own Edit tool schema requires
// (file_path/old_string/new_string). An audit caught that the repair logic
// below only ever rewrote the VALUE under whichever key the model used,
// never the key itself — a model that called Edit with {"path":...,
// "old_str":...,"new_str":...} got back the same alias names in
// tool_use.input, which Claude Code's Zod-schema-validated tool dispatcher
// rejects outright, since it requires file_path/old_string/new_string
// exactly.
func RepairEditToolCall(toolName string, argsRaw string, sessionID string) (string, bool) {
	repaired, changed := repairEditToolCallCore(toolName, argsRaw, sessionID)
	normalized, renamed := normalizeEditToolCallKeys(toolName, repaired)
	if renamed {
		return normalized, true
	}
	return repaired, changed
}

// normalizeEditToolCallKeys renames any alias key present to its canonical
// Claude Code Edit-tool name, deleting the alias. Safe to call even when no
// repair happened — it only touches key names, never values.
func normalizeEditToolCallKeys(toolName string, argsRaw string) (string, bool) {
	// NotebookEdit's real Claude Code schema uses notebook_path/new_source,
	// not file_path/old_string/new_string — it's excluded here (unlike
	// repairEditToolCallCore's broader alias-reading above, which is safe
	// either way since it only reads a value, never renames a key) because
	// renaming its genuinely-correct "notebook_path" to "file_path" would
	// break it, not fix it.
	nameLower := strings.ToLower(toolName)
	// An audit caught that renaming keys for "str_replace_editor" here
	// was actively wrong: that's Anthropic's own OFFICIAL built-in
	// text-editor tool name (the convention a real Anthropic API client
	// — not Claude Code, which has a completely different tool of its
	// own called "Edit" — uses when declaring text_editor_* without a
	// custom name). Its REAL canonical argument names are command/path/
	// old_str/new_str (builtinAnthropicToolSchema declares exactly
	// these), and a genuine Anthropic SDK client's own schema validation
	// expects path/old_str/new_str back unchanged — renaming them to
	// file_path/old_string/new_string broke a client using the tool
	// exactly as officially specified. Only Claude Code's own "Edit" (and
	// the generic MCP "update" convention) actually need this rename;
	// str_replace_editor's args still benefit from the broader VALUE
	// repair elsewhere in this file (whitespace/indentation fixing),
	// which reads either naming convention without ever renaming a key.
	isEditLike := nameLower == "edit" || nameLower == "update"
	// A follow-up audit caught that Write and Read — real Claude Code
	// tools needing the exact same file_path key, just with no old_string/
	// new_string concept at all — got NO alias normalization whatsoever,
	// the same "path" confusion already fixed for Edit. (Read has no
	// "content" input parameter to alias at all — the audit's claimed
	// "text -> content for Read" doesn't correspond to a real field —
	// so only file_path is normalized for it, same as Write.)
	isWriteOrRead := nameLower == "write" || nameLower == "read"
	if !isEditLike && !isWriteOrRead {
		return argsRaw, false
	}
	parsed := gjson.Parse(argsRaw)
	if !parsed.IsObject() {
		return argsRaw, false
	}
	renamed := false
	rename := func(canonical string, aliases []string) {
		if parsed.Get(canonical).Exists() {
			return
		}
		for _, alias := range aliases {
			if val := parsed.Get(alias); val.Exists() {
				argsRaw, _ = sjson.SetRaw(argsRaw, canonical, val.Raw)
				argsRaw, _ = sjson.Delete(argsRaw, alias)
				renamed = true
				return
			}
		}
	}
	rename("file_path", []string{"path", "notebook_path", "target_file"})
	if isEditLike {
		rename("old_string", []string{"old_str", "old_text", "pattern"})
		rename("new_string", []string{"new_str", "new_text"})
	}
	if nameLower == "write" {
		// Write's real "content" field is plausibly confused with "text",
		// the same way models confuse "content"/"text" for tool_result
		// bodies elsewhere in this gateway's own translation layer.
		rename("content", []string{"text"})
	}
	return argsRaw, renamed
}

func repairEditToolCallCore(toolName string, argsRaw string, sessionID string) (string, bool) {
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
	cwdCandidate := ""
	resolvedPath := targetPath
	if !isAbsPath(resolvedPath) {
		cwd := GetSessionCWD(sessionID)
		if cwd != "" {
			cwdCandidate = joinCWD(cwd, resolvedPath)
			if _, err := os.Stat(cwdCandidate); err == nil {
				resolvedPath = cwdCandidate
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
		if !ok && cwdCandidate != "" {
			snap, ok = getSessionSnapshot(sessionID, cwdCandidate)
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

	// An audit caught that stripCatNPrefix (added so a model copying its
	// own prior Read-tool output, including the "cat -n"-style "   42\t"
	// line-number prefix Read's dump carries, could still match) used to
	// run UNCONDITIONALLY before this exact-match check even ran — its
	// regex (`^\s*\d+\t`, digits then a literal tab) has no way to tell
	// a genuine Read-dump artifact apart from a real data line that just
	// happens to start the same way (a TSV row "1\tactive\ttrue", a
	// Makefile target index, ...). Mutating old_string BEFORE checking
	// whether the ORIGINAL already matched meant a real data line's
	// exact match was destroyed before ever being tried. Only attempt
	// the strip now, as a fallback, after the original has already
	// failed to match on its own.
	catNCleaned := false
	if cleaned := stripCatNPrefix(oldStr); cleaned != oldStr {
		if strings.Contains(fileContent, cleaned) {
			oldStr = cleaned
			catNCleaned = true
			if withClean, err := sjson.Set(argsRaw, oldStrKey, oldStr); err == nil {
				argsRaw = withClean
			}
			return argsRaw, true
		}
		// Didn't produce an exact match either, but may still help the
		// fuzzy matcher below (e.g. combined with tab/space repair) —
		// keep the cleaned version for the remainder of this function,
		// same as before.
		oldStr = cleaned
		catNCleaned = true
		if withClean, err := sjson.Set(argsRaw, oldStrKey, oldStr); err == nil {
			argsRaw = withClean
		}
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
	// An audit caught that gating the LF candidate on an else-branch of
	// hasCRLF (true the moment the file contains a SINGLE \r\n anywhere)
	// made it unreachable for a file that's mostly LF with one stray
	// CRLF elsewhere — the target section needing plain LF-normalization
	// never got tried at all. Try both candidates; a file can legitimately
	// contain both line-ending styles in different regions.
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
	}
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

	// 5. Advanced Whitespace-Insensitive & Tab Matching (Line-by-line Sliding Window)
	repairedExact, lineIndents, fileLine0Indent, ok := findFuzzyLineMatch(fileContent, oldStr)
	if ok && repairedExact != "" {
		repaired, err := sjson.Set(argsRaw, oldStrKey, repairedExact)
		if err == nil {
			// Ensure new_string matches CRLF/LF line endings of file
			if newStrKey != "" {
				effectiveNewStr := newStr
				if hasCRLF {
					newCRLFCandidate := strings.ReplaceAll(strings.ReplaceAll(newStr, "\r\n", "\n"), "\n", "\r\n")
					if newCRLFCandidate != newStr {
						effectiveNewStr = newCRLFCandidate
					}
				} else {
					newLFCandidate := strings.ReplaceAll(newStr, "\r\n", "\n")
					if newLFCandidate != newStr {
						effectiveNewStr = newLFCandidate
					}
				}
				// Re-apply the indentation findFuzzyLineMatch detected as
				// missing or mismatched on any line, including line 0 when
				// old_string's own line 0 had SOME indentation of its own
				// that didn't match the file's real indentation (a
				// zero-indentation line 0 is handled earlier via
				// repairedExact's start-offset shift instead). When
				// new_string's line count differs from old_string's (a
				// line added/removed), reapplyLineIndents falls back to a
				// uniform best-effort indent for lines 1..N instead of a
				// precise 1:1 mapping.
				if reindented, did := reapplyLineIndents(effectiveNewStr, lineIndents, fileLine0Indent, oldStr); did {
					effectiveNewStr = reindented
				}
				if effectiveNewStr != newStr {
					if withNew, err2 := sjson.Set(repaired, newStrKey, effectiveNewStr); err2 == nil {
						repaired = withNew
					}
				}
			}
			LogGateway("[Auto-Repair Edit] Successfully fuzzy-repaired old_string for %s (%d -> %d bytes, tabs/spaces matched)",
				filepath.Base(resolvedPath), len(oldStr), len(repairedExact))
			return repaired, true
		}
	}

	return argsRaw, catNCleaned
}

// reapplyLineIndents prepends lineIndents[j] onto newStr's line j, for every
// j where lineIndents[j] is non-empty and that line doesn't already start
// with whitespace (the model may have gotten SOME of its new lines right).
// Only applies when newStr has exactly as many lines as lineIndents has
// entries — a differing line count means old_string and new_string don't
// correspond 1:1 (lines were added/removed), and guessing which new line
// maps to which old one would risk indenting the wrong line instead of
// just leaving the ambiguous case alone.
// stripOnlyIndentMarker is a sentinel findFuzzyLineMatch stores in
// lineIndents[0] instead of "" when its own start-offset shift already
// preserved the file's real indentation as untouched surrounding text —
// new_string's line 0 must have any leading whitespace of its OWN
// stripped (not replaced with anything) so it doesn't stack on top of
// that preserved prefix, but nothing should be prepended here.
const stripOnlyIndentMarker = "\x00strip-only\x00"

func reapplyLineIndents(newStr string, lineIndents []string, fileLine0Indent string, oldStr string) (string, bool) {
	hasAny := false
	for _, ind := range lineIndents {
		if ind != "" {
			hasAny = true
			break
		}
	}
	// An audit caught that a single-line old_string expanding into a
	// multi-line new_string had no information at all to indent lines
	// 1..N with: there's no old_string line 1..N to compare against the
	// file, so lineIndents (length 1) carries nothing for them even when
	// old_string's own line 0 matched the file exactly (lineIndents[0] ==
	// ""). Keep going in that specific case so the fallback below can use
	// fileLine0Indent as a best-effort base for the newly-inserted lines,
	// instead of leaving them flush-left.
	if !hasAny && !(len(lineIndents) == 1 && fileLine0Indent != "" && strings.Contains(newStr, "\n")) {
		return newStr, false
	}
	lines := strings.Split(newStr, "\n")
	// An audit caught that findFuzzyLineMatch strips a trailing newline's
	// resulting empty element from old_string's own line split (giving
	// lineIndents a length matching the REAL line count), but this
	// function's own split never got the same treatment — a completely
	// normal new_string that happens to end in "\n" with the exact same
	// real line count as old_string had ONE MORE element here than
	// lineIndents, so the precise exact-1:1 branch below never fired for
	// what should be its most common, simplest case, falling through to
	// the less-precise heuristic branch instead. Strip it the same way,
	// and remember to restore it when rejoining at the end.
	hadTrailingNewline := false
	if len(lines) > 1 && lines[len(lines)-1] == "" && strings.HasSuffix(newStr, "\n") {
		lines = lines[:len(lines)-1]
		hadTrailingNewline = true
	}
	changed := false

	// Captured before line 0 is potentially rewritten below — the
	// model's OWN relative indentation between line 0 and later lines
	// (e.g. line 0 has none of its own, line 3 has 4 extra spaces of its
	// own) is real signal about nesting depth that a flat "apply one
	// uniform indent to every line" replacement throws away. See its use
	// in the differing-line-count branch below.
	origLine0OwnIndent := lines[0][:len(lines[0])-len(strings.TrimLeft(lines[0], " \t"))]

	// Line 0 is handled the same way regardless of whether the rest of
	// the lines correspond 1:1: lineIndents[0] already holds the exact
	// target indentation whenever anything is missing there (findFuzzyLineMatch
	// zeroes it out itself when its own start-offset shift already fixed
	// it), so there's never a need to guess at a uniform indent for it the
	// way lines 1..N do below.
	if len(lineIndents) > 0 && lineIndents[0] == stripOnlyIndentMarker {
		trimmed := strings.TrimLeft(lines[0], " \t")
		if trimmed != lines[0] {
			lines[0] = trimmed
			changed = true
		}
	} else if len(lineIndents) > 0 && lineIndents[0] != "" {
		trimmed := strings.TrimLeft(lines[0], " \t")
		if lineIndents[0]+trimmed != lines[0] {
			lines[0] = lineIndents[0] + trimmed
			changed = true
		}
	}

	if len(lines) == len(lineIndents) {
		// Exact 1:1 correspondence — reapply each line's own detected indent.
		// lineIndents[j] is the file's real indentation whenever it differs
		// from old_string's own line j (findFuzzyLineMatch already filters
		// out lines where they match), so replace whatever leading
		// whitespace new_string's line carries — not just fill in an
		// absent one — to also correct a wrong style/depth (e.g. the model
		// mirrored its own space-indented old_string against a
		// tab-indented file).
		for j, ind := range lineIndents {
			if ind == "" || j == 0 {
				continue // line 0 already handled above
			}
			trimmed := strings.TrimLeft(lines[j], " \t")
			if ind+trimmed == lines[j] {
				continue // already correct
			}
			lines[j] = ind + trimmed
			changed = true
		}
	} else {
		// An audit caught that a differing line count (the model added or
		// removed a line — common when editing, not an edge case) made
		// this bail out entirely, leaving EVERY line past the first
		// flush-left with no indentation at all — worse than doing
		// nothing targeted, since this is the common case, not the rare
		// one. No 1:1 line mapping is possible here, but old_string's own
		// lines overwhelmingly tend to share one indentation depth (a
		// single block/function body) — apply the most common non-empty
		// detected indent uniformly to every new_string line (other than
		// line 0, handled above) that doesn't already have its own
		// leading whitespace, instead of giving up and guaranteeing
		// flush-left on all of them.
		counts := make(map[string]int)
		for j, ind := range lineIndents {
			if j == 0 || ind == "" {
				continue
			}
			counts[ind]++
		}
		var baseIndent string
		best := 0
		for ind, n := range counts {
			if n > best {
				best = n
				baseIndent = ind
			}
		}
		if len(lineIndents) == 1 && fileLine0Indent != "" {
			// old_string was a single line, so there's no real lines-1..N
			// detected indentation at all to pick a baseIndent from
			// (counts is necessarily empty) — this is the ONLY case
			// handled by relative-delta preservation below, specifically
			// because it's the only case with no per-line ground truth to
			// prefer instead. An audit caught that naively using
			// fileLine0Indent as a flat baseIndent here flattens nested
			// structure: expanding "def foo():" into
			// "def foo():\n    return 42" (new_string's own text already
			// correctly shows line 1 four spaces DEEPER than its own line
			// 0 baseline) lost that relative nesting and put both lines
			// at the identical depth, corrupting Python's indentation
			// rules. Layer new_string's own relative indentation (vs. its
			// own line 0) on top of the file's real indentation instead
			// of discarding it.
			for j := range lines {
				if j == 0 {
					continue // line 0 already handled above
				}
				if strings.TrimSpace(lines[j]) == "" {
					continue // don't indent a genuinely blank line
				}
				ownLeadingLen := len(lines[j]) - len(strings.TrimLeft(lines[j], " \t"))
				ownIndent := lines[j][:ownLeadingLen]
				trimmed := lines[j][ownLeadingLen:]
				var target string
				if strings.HasPrefix(ownIndent, origLine0OwnIndent) {
					delta := ownIndent[len(origLine0OwnIndent):]
					target = fileLine0Indent + delta + trimmed
				} else {
					// Own indentation doesn't extend line 0's baseline
					// (e.g. an actual dedent) — a prefix-delta can't
					// express that; best-effort flat fallback.
					target = fileLine0Indent + trimmed
				}
				if target == lines[j] {
					continue // already correct
				}
				lines[j] = target
				changed = true
			}
		} else if baseIndent != "" {
			// An audit caught that applying the SAME flat baseIndent to
			// every line here flattens genuinely nested structure within
			// the replacement itself: old_string spanning a for-loop with
			// an if nested inside it has lines at 3 different real
			// depths, but adding one new line inside that if (changing
			// the line count, which is why we're in this branch at all)
			// collapsed the for-header, the if-header, the body, AND
			// both closing braces onto the SAME single depth — not just
			// at the line-0 boundary (the narrower case fixed above), but
			// throughout the whole block.
			//
			// old_string's own lines (split the same way
			// findFuzzyLineMatch did, to stay index-aligned with
			// lineIndents) already carry the REAL per-line file
			// indentation for any line whose own TEXT is unchanged in
			// new_string. A plain content->indent map collides on
			// duplicate lines (two bare "}" closing braces at DIFFERENT
			// real depths hash to the same key, so the second overwrites
			// the first) — match them in ORDER instead, like a simple
			// diff: walk old_string's lines once, and for each one scan
			// new_string forward from just past the previous match to
			// find its next occurrence, assigning that old line's real
			// indent to that exact position. Lines in new_string with no
			// match at all (walked past without ever being found) are
			// genuinely new insertions with no ground truth, and still
			// fall back to baseIndent.
			oldLines := strings.Split(strings.ReplaceAll(oldStr, "\r\n", "\n"), "\n")
			if (strings.HasSuffix(oldStr, "\n") || strings.HasSuffix(oldStr, "\r")) && len(oldLines) > 1 && oldLines[len(oldLines)-1] == "" {
				oldLines = oldLines[:len(oldLines)-1]
			}
			assignedIndent := make([]string, len(lines))
			assigned := make([]bool, len(lines))
			searchFrom := 1
			for k := 1; k < len(oldLines) && k < len(lineIndents); k++ {
				oldOwnLeadingLen := len(oldLines[k]) - len(strings.TrimLeft(oldLines[k], " \t"))
				oldContent := normalizeLineForMatch(oldLines[k][oldOwnLeadingLen:])
				for j := searchFrom; j < len(lines); j++ {
					if normalizeLineForMatch(strings.TrimLeft(lines[j], " \t")) != oldContent {
						continue
					}
					realIndent := lineIndents[k]
					if realIndent == "" {
						// Nothing was missing for this old line, meaning
						// its own leading whitespace already equals the
						// file's — that own whitespace IS the real indent.
						realIndent = oldLines[k][:oldOwnLeadingLen]
					}
					assignedIndent[j] = realIndent
					assigned[j] = true
					searchFrom = j + 1
					break
				}
			}
			// A genuinely new (unmatched) line most likely sits right next
			// to a matched one in the model's own intended structure (a
			// new statement inserted after doA() is almost certainly a
			// sibling of doA(), not sitting at the enclosing block's
			// average depth) — prefer the nearest PRECEDING matched
			// line's real indent over the generic baseIndent guess when
			// one is available.
			//
			// An audit caught that gluing the nearest preceding anchor's
			// indent on flatly (after stripping the new line's OWN
			// leading whitespace) threw away real signal when the model
			// wrote the new line with its own correct, DEEPER relative
			// indent than that anchor — e.g. inserting a statement one
			// level further nested than the line right before it still
			// landed at the anchor's flat depth, not one level deeper.
			// Snapshot each line's own ORIGINAL leading whitespace before
			// any mutation below, then — same technique already used for
			// the single-old-line case above — compute the new line's
			// indentation delta relative to the anchor's own original
			// whitespace, and layer that delta on top of the anchor's
			// REAL (corrected) indent, instead of discarding it. The
			// anchor starts as line 0 itself (own indent / corrected
			// indent) so even an insertion before the first matched line
			// benefits from this, not just insertions after one.
			originalOwnIndent := make([]string, len(lines))
			for j, l := range lines {
				originalOwnIndent[j] = l[:len(l)-len(strings.TrimLeft(l, " \t"))]
			}
			anchorOwnIndent := origLine0OwnIndent
			anchorRealIndent := fileLine0Indent
			for j := range lines {
				if j == 0 {
					continue // line 0 already handled above
				}
				if assigned[j] {
					anchorOwnIndent = originalOwnIndent[j]
					anchorRealIndent = assignedIndent[j]
				}
				if strings.TrimSpace(lines[j]) == "" {
					continue // don't indent a genuinely blank line
				}
				trimmed := strings.TrimLeft(lines[j], " \t")
				var target string
				switch {
				case assigned[j]:
					target = assignedIndent[j] + trimmed
				case anchorRealIndent != "" && strings.HasPrefix(originalOwnIndent[j], anchorOwnIndent):
					delta := originalOwnIndent[j][len(anchorOwnIndent):]
					target = anchorRealIndent + delta + trimmed
				case baseIndent != "":
					target = baseIndent + trimmed
				default:
					target = anchorRealIndent + trimmed
				}
				if target == lines[j] {
					continue // already correct
				}
				lines[j] = target
				changed = true
			}
		}
	}
	if !changed {
		return newStr, false
	}
	result := strings.Join(lines, "\n")
	if hadTrailingNewline {
		result += "\n"
	}
	return result, true
}

// findFuzzyLineMatch returns (the real file content matching oldStr,
// per-line leading-whitespace that was stripped from the comparison and
// must be re-applied to new_string if the caller doesn't want to lose it,
// ok). lineIndents has exactly as many entries as oldStr has lines; most
// are "" (that line's own indentation already matched).
func findFuzzyLineMatch(fileContent, oldStr string) (string, []string, string, bool) {
	// Parse model lines
	rawOldLines := strings.Split(strings.ReplaceAll(oldStr, "\r\n", "\n"), "\n")
	hasTrailingNewline := strings.HasSuffix(oldStr, "\n") || strings.HasSuffix(oldStr, "\r")
	if hasTrailingNewline && len(rawOldLines) > 1 && rawOldLines[len(rawOldLines)-1] == "" {
		rawOldLines = rawOldLines[:len(rawOldLines)-1]
	}
	if len(rawOldLines) == 0 {
		return "", nil, "", false
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
		return "", nil, "", false
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

		// An audit caught that the line-0 fix below (restoring the file's
		// real leading whitespace so old_string matches byte-for-byte) was
		// never extended to lines 1..N: new_string's matching lines keep
		// whatever indentation the model wrote them with — typically none,
		// mirroring its own flawed old_string — so the replace strips the
		// file's real indentation from every line after the first,
		// corrupting the result (especially syntactically significant for
		// Python/YAML). Record each line's missing indentation here so the
		// caller can re-apply it to new_string.
		// An audit further caught that checking "old_string's line already
		// had SOME leading whitespace" (any space/tab) was too weak: a
		// model that writes old_string with its own space-indentation
		// against a tab-indented file passes that check and gets treated
		// as "nothing missing", even though the style/depth is wrong — the
		// replacement then injects space-indented lines into a tab file
		// (or vice versa), corrupting Go/Python/YAML indentation rules
		// just as badly as missing indentation does. Compare against the
		// file's actual leading whitespace for that line, not just
		// presence vs. absence, so a mismatched style/depth is corrected
		// too, not only a totally absent one.
		lineIndents := make([]string, mLen)
		for j := 0; j < mLen; j++ {
			fileLineClean := fileLines[startIdx+j].clean
			fileLeadingLen := len(fileLineClean) - len(strings.TrimLeft(fileLineClean, " \t"))
			fileIndent := fileLineClean[:fileLeadingLen]
			oldLeadingLen := len(rawOldLines[j]) - len(strings.TrimLeft(rawOldLines[j], " \t"))
			oldIndent := rawOldLines[j][:oldLeadingLen]
			if fileIndent == oldIndent {
				continue // old_string's own line j already matches the file's real indentation; nothing missing
			}
			lineIndents[j] = fileIndent
		}
		// Captured unconditionally (unlike lineIndents[0], which stays ""
		// when old_string's line 0 already matched) so the caller has the
		// file's real line-0 indentation available even in that matching
		// case — needed below for a single-line-to-multi-line replacement,
		// where lines 1..N never existed in old_string at all and so have
		// no indent of their own to detect a mismatch against.
		fileLine0Clean := fileLines[startIdx].clean
		fileLine0Indent := fileLine0Clean[:len(fileLine0Clean)-len(strings.TrimLeft(fileLine0Clean, " \t"))]

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
		//
		// An audit caught that this shift also fired when old_string's
		// line 0 DID have its own (wrong-style/wrong-depth) leading
		// whitespace — e.g. 4 spaces against a tab-indented file. In that
		// case lineIndents[0] holds the file's real tab, and shifting
		// startOffset past it excludes the real tab from the matched span
		// entirely, leaving it as untouched surrounding text; new_string's
		// line 0 (still carrying its own 4 spaces, since reapplyLineIndents
		// skipped line 0 unconditionally) then gets inserted right after
		// that preserved tab, producing "\t    " — tab AND spaces both
		// present. Only take this shortcut when old_string's own line 0
		// had NO leading whitespace at all; otherwise leave the real
		// indentation inside the matched span (so it gets replaced, not
		// preserved-and-duplicated) and let reapplyLineIndents correct
		// new_string's line 0 the same way it corrects every other line.
		oldLine0LeadingLen := len(rawOldLines[0]) - len(strings.TrimLeft(rawOldLines[0], " \t"))
		if lineIndents[0] != "" && oldLine0LeadingLen == 0 {
			startOffset += len(lineIndents[0])
			// An audit caught a remaining edge case even after the shift:
			// it assumes new_string's line 0 mirrors old_string's own lack
			// of indentation (the common case), but if the model happened
			// to write new_string's line 0 WITH its own leading whitespace
			// anyway (while old_string had none), that whitespace would
			// land right after the real indentation this shift already
			// preserves, doubling it (e.g. "\t\t" instead of "\t"). Use a
			// sentinel so reapplyLineIndents strips any such whitespace
			// from new_string's line 0 without prepending anything — the
			// correct indentation already comes from the preserved span.
			lineIndents[0] = stripOnlyIndentMarker
		}
		lastLine := fileLines[endIdx]

		// If the last line in model's old_string did not end with a newline,
		// strip trailing newline from the matched file segment to match model's boundary intention
		endOffset := lastLine.offset + len(lastLine.text)
		if !strings.HasSuffix(oldStr, "\n") && !strings.HasSuffix(oldStr, "\r") {
			endOffset = lastLine.offset + len(lastLine.clean)
		}

		if endOffset <= len(fileContent) && startOffset <= endOffset {
			return fileContent[startOffset:endOffset], lineIndents, fileLine0Indent, true
		}
	}

	return "", nil, "", false
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
	// An audit caught that a model calling Grep with {"path":"main.go",
	// "query":"HandleRequest"} (a plausible alias a weaker model reaches
	// for) fell straight into the "path looks like it's actually the
	// search term" heuristic below — "main.go" has no special regex
	// chars AND no slash, so it passed — repurposing the FILENAME as the
	// search pattern, deleting "path" entirely, and silently discarding
	// "query" (the model's real intended search term). Check for an
	// actual pattern-alias key first; only fall back to repurposing path
	// when none of those exist either.
	for _, aliasKey := range []string{"query", "search", "regex", "term", "text"} {
		if alias := parsed.Get(aliasKey).String(); alias != "" {
			fixed, _ := sjson.Set(argsRaw, "pattern", alias)
			fixed, _ = sjson.Delete(fixed, aliasKey)
			return fixed, true
		}
	}
	// An audit caught that "\\" in the first check (regex special chars)
	// falsely matched ANY Windows-style path at all — "\" is also a
	// literal path separator there, and "controllers\auth.go" has no
	// genuine regex metacharacter, just a normal Windows directory
	// separator. Keep "\\" in the SECOND check (no separator at all —
	// still correctly recognizes "\" as a valid separator there, so a
	// real Windows path doesn't also trip that one), but drop it from
	// the regex-metacharacter set.
	if path != "" && (strings.ContainsAny(path, "*?[]()|^$+") || !strings.ContainsAny(path, "/\\")) {
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
	// An audit caught that, unlike RepairGrepToolCall just above, this had
	// no alias-key fallback at all — a model calling Glob(glob:"*.go")
	// (an entirely plausible alias given the tool's own name) left
	// "pattern" missing, the required-parameter validation on Claude
	// Code's side rejecting the call outright.
	for _, aliasKey := range []string{"glob", "query", "search"} {
		if alias := parsed.Get(aliasKey).String(); alias != "" {
			fixed, _ := sjson.Set(argsRaw, "pattern", alias)
			fixed, _ = sjson.Delete(fixed, aliasKey)
			return fixed, true
		}
	}
	if path != "" && strings.ContainsAny(path, "*?[]") {
		fixed, _ := sjson.Set(argsRaw, "pattern", path)
		fixed, _ = sjson.Delete(fixed, "path")
		return fixed, true
	}
	return argsRaw, false
}
