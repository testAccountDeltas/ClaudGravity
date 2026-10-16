package main

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"unicode"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// SessionWorkingDirectory tracks the active working directory discovered per session or agent
var (
	sessionCWDLock sync.RWMutex
	sessionCWDMap  = make(map[string]string)
)

func RecordSessionCWD(sessionID, cwd string) {
	if sessionID == "" || cwd == "" {
		return
	}
	sessionCWDLock.Lock()
	defer sessionCWDLock.Unlock()
	sessionCWDMap[sessionID] = filepath.Clean(strings.TrimSpace(cwd))
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

// ExtractCWDFromSystemPrompt extracts the user session working directory from Claude Code environment preamble
func ExtractCWDFromSystemPrompt(systemPrompt string) string {
	if systemPrompt == "" {
		return ""
	}
	re := regexp.MustCompile(`(?:Primary working directory|Current working directory|working directory)[:\s]+([^\r\n]+)`)
	m := re.FindStringSubmatch(systemPrompt)
	if len(m) > 1 {
		p := strings.TrimSpace(m[1])
		p = strings.Trim(p, "\"'`")
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
	if !isWithinSessionCWD(resolvedPath, sessionID) {
		return argsRaw, false
	}

	data, err := os.ReadFile(resolvedPath)
	if err != nil {
		// File does not exist or unreadable, cannot repair
		return argsRaw, false
	}

	fileContent := string(data)

	// Direct byte-for-byte match already exists! No repair needed.
	if strings.Contains(fileContent, oldStr) {
		return argsRaw, false
	}

	// 4. Try newline normalization (\r\n vs \n)
	hasCRLF := strings.Contains(fileContent, "\r\n")
	normalizedLF := strings.ReplaceAll(oldStr, "\r\n", "\n")
	if hasCRLF {
		crlfCandidate := strings.ReplaceAll(normalizedLF, "\n", "\r\n")
		if strings.Contains(fileContent, crlfCandidate) {
			repaired, err := sjson.Set(argsRaw, oldStrKey, crlfCandidate)
			if err == nil {
				LogGateway("[Auto-Repair Edit] Fixed CRLF newlines for %s (len: %d)", filepath.Base(resolvedPath), len(crlfCandidate))
				return repaired, true
			}
		}
	} else {
		if strings.Contains(fileContent, normalizedLF) {
			repaired, err := sjson.Set(argsRaw, oldStrKey, normalizedLF)
			if err == nil {
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
