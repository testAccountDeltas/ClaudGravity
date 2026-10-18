package main

import (
	"crypto/sha256"
	"encoding/hex"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/tidwall/gjson"
)

const (
	// ReadWindowSize/ReadWindowThreshold replace a strict "N consecutive
	// pure-Read turns" counter: measured live (84 subagent requests, one
	// real Explore session), Gemini rarely does 10 single-Read turns in an
	// unbroken row — it interleaves Bash or batches 2-4 Reads per turn — so
	// the old streak-based counter never fired even with 44 total Read
	// calls scattered across the session. This instead sums Read calls
	// over a sliding window of recent turns regardless of what's
	// interspersed, and only resets on genuine progress (Write/Edit, or a
	// substantial explanation) rather than on every non-Read turn.
	ReadWindowSize      = 15
	ReadWindowThreshold = 10
	ToolCallHistMax     = 12
	OscIgnoreTag        = "[ignoring loop detection]"
	// Recommends Bash (grep/rg), not the native Grep tool — strip_search_tools
	// defaults to true (see config.go), so Grep/Glob usually aren't even
	// declared to the model; telling it to use a tool it doesn't have is
	// useless advice at best and confusing at worst.
	NudgeReadLoop    = "Notice from the gateway: you have made many Read calls recently without writing/editing anything or giving a substantial answer — this looks like an unproductive read-loop. Stop reading slices. Use Bash (grep/rg) to jump straight to the relevant symbol/section, or read larger ranges, or proceed with what you have."
	NudgeOscillation = "Notice from the gateway: your output is flagged for looping content (the exact same tool call or A/B oscillation repeated). If intentional, add the tag [ignoring loop detection]; otherwise stop and try a different approach."

	// readToolGuidance is appended to the Read tool's own declared
	// "description" field (translator.go, where Claude's tool
	// declarations get translated into Gemini functionDeclarations) —
	// not the system prompt. Two earlier versions lived in the system
	// prompt instead and both measurably helped (fixed ~120-line pages
	// became variable 40-150-line pages, then a direct test got a
	// correctly-reasoned no-offset/limit call) without fully stopping
	// the pattern on real multi-file audits. Moved to the tool's own
	// description because: (1) it sits right next to the schema the
	// model is already attending to when deciding how to call this
	// specific tool, instead of competing with everything else in the
	// system prompt; (2) it's part of the tool DEFINITION, sent fresh
	// with the same static text on every request, same caching
	// properties as the system-prompt version had — never put a date,
	// counter, or anything else that varies per-request in here, or it
	// breaks Gemini's implicit prefix cache for no benefit.
	//
	// Three things wrong with the system-prompt version, fixed here:
	// (1) a dense file (long Go struct/comment lines) can hit the Read
	// tool's own OUTPUT TOKEN cap before 2000 LINES, forcing a real
	// "file too large" retry — the old text had no guidance for that
	// case, so the model fell back to its own habit of small ~100-line
	// pages. Now explicit: retry with a LARGE chunk (500-800 lines),
	// never a small one. (2) the old text's "the whole file ends up in
	// context either way" framing only holds when the whole file is
	// actually needed — now split out as its own bullet, since reading
	// one specific function should go through Grep first, not a
	// full-file read. (3) "follow this exactly" plus a paragraph of
	// justification was wordier than this model needs — shorter, with
	// concrete numbers, reads as more authoritative.
	// v4 adds three short pieces, all from an audit of this exact text: a
	// safety-reassurance line (using only the one fact independently
	// verified — the tool's own description already says "by default
	// reads up to 2000 lines" — NOT the audit's unverified decompiled
	// byte/token-cap numbers from claude.exe, which stay unconfirmed and
	// are deliberately left out of this text); a line closing off the
	// shell-trick workaround path at the source instead of only
	// reactively detecting it (readLoopEquivalentToolName above); and an
	// explicit "check first, don't guess" procedure — the model's own
	// stated reason for defaulting to small pages is that it doesn't
	// know a file's size before reading it, so give it the concrete
	// first step (list the directory / check size) instead of just
	// telling it not to guess.
	readToolGuidance = "Gateway notice (Read tool): Do not read files in small pages. Call Read without offset/limit by default: it returns up to 2000 lines in one call, which covers most files, and the tool itself safely handles anything too large — you do not need to protect against that yourself. Splitting into 100-150 line pages wastes calls and time and saves no context, because the whole file ends up in context anyway.\n\nIf this conversation already showed this file's size — an earlier ls -la, find, or directory listing — use that number, don't default to a cautious small chunk out of habit when you already know the answer. If you don't know the size at all, check first (ls -la) instead of guessing. Under ~150KB: always read it whole, one call, no offset/limit. That covers the large majority of individual source files.\n\nUse offset/limit only when:\n- the file is confirmed at or above that size (then use big chunks, at least 500-800 lines, never 100-150), or\n- a Grep/Bash search already gave you the exact line range you need, and you only need that range.\n\nTo locate one function or symbol in a large file, Grep first, then Read just that range. Do not scan the file page by page. Switching to sed/awk/seq/head/tail/cat/python one-liners to page through a file the same way does not help — it is the same pattern under a different tool and is tracked the same way."

	// NudgeEditFailureLoop fires when Edit has failed (old_string mismatch)
	// several times in a row for the same agent — the read-loop guard above
	// has no visibility into this at all, since whether an Edit succeeded
	// only becomes known on the client's NEXT request (its tool_result's
	// is_error), not in this gateway's response to the call that failed.
	NudgeEditFailureLoop = "Notice from the gateway: your Edit tool calls have failed repeatedly because old_string did not match the file. Do not keep retrying minor variations of the same old_string. Read the file to see its exact current contents, then either fix old_string to match exactly or use Write to replace the whole section/file."

	// NudgeReadLoopEscalated fires when the soft nudge above already fired
	// once for this agent and it slid right back into another full
	// read-loop window without ever writing/editing anything in between —
	// confirmed real-world pattern: a model politely acknowledges the soft
	// nudge ("Understood, I'll proceed") and then immediately reads again.
	NudgeReadLoopEscalated = "Notice from the gateway (repeated): you were already warned about an unproductive read-loop and went straight back into another one without writing, editing, or answering in between. Do not read anything else right now. In your very next turn, either call Write/Edit to make the change, or give your final answer with what you already have."

	// loopGuardStaleAfter/loopGuardSweepEvery bound the otherwise-unbounded
	// growth of readWindows/toolCallHist/lastSeen across the many distinct,
	// ephemeral agentKeys a long-running gateway process accumulates.
	loopGuardStaleAfter = 2 * time.Hour
	loopGuardSweepEvery = 30 * time.Minute

	// exploreReadWindowSize/exploreReadWindowThreshold are the read-loop
	// limits for any subagent role instead of the main-agent
	// ReadWindowSize/ReadWindowThreshold above. Originally scoped to just
	// Explore/Plan, but "general-purpose" (the most commonly spawned
	// subagent — the Agent tool's own catch-all for "researching complex
	// questions, searching for code, and executing multi-step tasks") is
	// built for the exact same kind of long, unsupervised read-heavy
	// investigation and was still hitting the main agent's tight
	// 10-reads-in-15-turns threshold and getting cut off mid-investigation.
	exploreReadWindowSize      = 50
	exploreReadWindowThreshold = 40
)

// readLoopLimits picks the LoopGuard read-loop window size and trip
// threshold for this turn's agent. agentKey is ParseAgentInfo's own output
// ("sub:<agentType>" for any subagent), so this needs no extra plumbing —
// the role is already encoded in the same string LoopGuard keys its state
// by. If hasModifyingTools is false (e.g. Plan Mode, or an investigation agent
// without edit permissions), we use the relaxed exploreReadWindow limits so
// legitimate file inspection does not trigger false-positive read-loop warnings.
// readEquivalentBashPattern matches a Bash command doing the same "dump
// raw file content" work the Read tool does — sed -n, head, tail, cat,
// seq-driven range loops, dd/cut/fold/nl — as opposed to a genuinely
// targeted search (grep/rg, or awk used as a pattern search), which is
// exactly what NudgeReadLoop itself tells the model to switch to and must
// never count against this same guard, or the nudge's own advice would
// trip it again immediately.
//
// awk is handled separately below, not in this pattern: an audit caught
// that the original version of this check excluded "awk" unconditionally
// (treating it as always a search), which left `awk 'NR>=1 && NR<=120'` —
// functionally identical line-range pagination, not a search —
// completely invisible to the counter. awkNRPattern distinguishes the
// two: an "NR <comparison>" anywhere in the script is pagination; an awk
// call with no NR comparison at all (e.g. `awk '/pattern/'`) is a real
// search and must stay excluded.
// Head/tail now also match the flagless BSD-style form (`head -50 file`,
// `tail +10 file`) and bare `more`/`less`/`pr` pagers — an audit caught
// that the old `-[nc]`-only requirement missed these entirely. `cat` stays
// in the group (bare `cat file` is still a read), but isReadEquivalentBashCommand
// below excludes it specifically when it's writing TO a file (heredoc/
// redirect), not reading one.
//
// A second audit caught that `head\s+(-[nc]|...)` still required SOME
// flag or number right after "head"/"tail" — a genuinely bare `head
// file.go` or `tail file.go` (no flags at all, the single most common way
// to type either command) matched neither alternative and was completely
// invisible to this guard. `cat`/`more`/`less`/`pr` already match bare
// with no such requirement; head/tail now do too, dropping the dangling
// \s+(...) group entirely.
var readEquivalentBashPattern = regexp.MustCompile(`(?i)\b(sed\s+-n|head|tail|cat|more|less|pr\b|seq\b|dd\s+bs=|cut\s+-[df]|fold\b|nl\s+-)\b`)
var awkNRPattern = regexp.MustCompile(`(?i)\bNR\s*[<>=!]`)

// catIsWritePattern matches `cat` used to WRITE a file (heredoc into a
// redirect, e.g. `cat << 'EOF' > file.go`, or `cat file1 file2 > out`) —
// an audit caught this counting as a Read when it's actually genuine
// progress (a file write), which should reset the window, not add to it.
var catIsWritePattern = regexp.MustCompile(`(?i)\bcat\b[^|]*>`)

// stderrRedirectPattern matches a file-descriptor-specific redirect used to
// suppress error output — `2>/dev/null`, `2>&1`, `2>$null` (PowerShell) —
// rather than a genuine stdout write. An audit caught that catIsWritePattern
// and powerShellIsWritePattern both just look for any literal ">" in the
// command, with no way to distinguish that from a plain READ command that
// merely silences its stderr (e.g. `cat file.txt 2>/dev/null`, `Get-Content
// file.txt 2>$null`) — RE2 (Go's regexp engine) has no lookbehind to assert
// "not preceded by a digit" directly in the write-pattern regexes
// themselves, so strip this specific shape out of the command first,
// before testing for a real write-redirect.
var stderrRedirectPattern = regexp.MustCompile(`[0-9]>\S*`)

func stripStderrRedirects(command string) string {
	return stderrRedirectPattern.ReplaceAllString(command, "")
}

// grepInvocationPattern requires grep/rg to appear as an actual COMMAND —
// at the start of the string, or right after a shell command-boundary
// (whitespace, pipe, semicolon, &&/||) — not merely as a substring anywhere
// in the command line. An audit caught that a plain strings.Contains(lc,
// "grep") also matched "grep" sitting inside a file PATH being read, e.g.
// `cat cmd/grep/main.go` or `less grep_test.go`, wrongly treating an
// ordinary file read as "used search" and skipping it from the read-loop
// counter entirely — the exact opposite of what a loop-nudge guard is for.
var grepInvocationPattern = regexp.MustCompile(`(?i)(^|[\s;&|])(grep|rg)\b`)

// psSearchInvocationPattern is grepInvocationPattern's PowerShell
// counterpart, covering findstr/grep/rg the same way — select-string is
// intentionally excluded here since it's checked separately with a plain
// substring match (a file path segment named exactly "select-string" is
// implausible enough not to bother with a command-boundary check).
var psSearchInvocationPattern = regexp.MustCompile(`(?i)(^|[\s;&|])(findstr|grep|rg)\b`)

func isReadEquivalentBashCommand(command string) bool {
	lc := strings.ToLower(command)
	if grepInvocationPattern.MatchString(command) {
		return false
	}
	if strings.Contains(lc, "awk") {
		return awkNRPattern.MatchString(command)
	}
	// python/perl one-liners doing the same slice-the-file trick under a
	// different interpreter — not exhaustive (this is inherently an
	// arms race the audit itself calls out), but covers the concrete
	// pattern actually observed/predicted: readlines()[a:b]-style slicing.
	if strings.Contains(lc, "readlines") || strings.Contains(lc, "splitlines") {
		return true
	}
	if strings.Contains(lc, "cat") && catIsWritePattern.MatchString(stripStderrRedirects(command)) {
		return false
	}
	return readEquivalentBashPattern.MatchString(command)
}

// isReadEquivalentPowerShellCommand is readEquivalentBashPattern's
// counterpart for the PowerShell tool — an audit caught that
// readLoopEquivalentToolName only ever recognized "Read" and "Bash" by
// name, leaving every PowerShell call (the primary shell tool on Windows,
// where this gateway's own maintainer runs it) completely invisible to
// the read-loop counter regardless of what it did. Select-String is
// PowerShell's grep equivalent — a genuine targeted search — and must
// stay excluded the same way grep/rg does for Bash.
// cat is a built-in PowerShell alias for Get-Content — an audit caught it
// was missing here, leaving a bare `cat file.txt` completely invisible.
var readEquivalentPowerShellPattern = regexp.MustCompile(`(?i)\b(get-content|gc|cat)\b|select-object\s+-(first|last|skip)|\btype\b|\bmore\b`)

// powerShellIsWritePattern is catIsWritePattern's PowerShell counterpart —
// an audit caught that `cat file.txt > out.txt` or `gc file.txt | Out-File
// out.txt` (reading one file specifically to write it elsewhere — genuine
// forward progress) had no exclusion here, miscounting it as an
// unproductive read-loop turn the same way plain Bash `cat` without this
// guard used to.
var powerShellIsWritePattern = regexp.MustCompile(`(?i)>|Out-File|Set-Content|Add-Content`)

func isReadEquivalentPowerShellCommand(command string) bool {
	lc := strings.ToLower(command)
	// select-string is PowerShell's own grep equivalent; findstr/rg/grep are
	// commonly invoked from PowerShell too (findstr is a native Windows
	// tool, rg/grep if Git-for-Windows or similar is on PATH) — an audit
	// caught that only select-string was excluded, so a genuine targeted
	// search piped through any of the others got miscounted as unproductive
	// reading instead of recognized as exactly what the nudge recommends.
	if strings.Contains(lc, "select-string") || psSearchInvocationPattern.MatchString(command) {
		return false
	}
	if readEquivalentPowerShellPattern.MatchString(command) {
		if powerShellIsWritePattern.MatchString(stripStderrRedirects(command)) {
			return false
		}
		return true
	}
	// An audit caught that isReadEquivalentBashCommand's own
	// readlines/splitlines check (for a Python/Perl one-liner invoked from
	// Bash) had no PowerShell counterpart — on Windows the exact same
	// dump-the-file one-liners are routinely invoked through `python -c
	// "..."`, `node -e "..."`, or .NET's own [System.IO.File] static
	// methods, all completely invisible to this guard without this.
	if strings.Contains(lc, "readlines") || strings.Contains(lc, "readfilesync") ||
		strings.Contains(lc, "readalllines") || strings.Contains(lc, "readalltext") {
		return true
	}
	return false
}

// isPlanModeRequest detects the same thing translator.go's own inline check
// does (ExitPlanMode declared among the request's tools) from raw request
// bytes, so server.go's RecordReadTurn call sites — which only ever have
// claudeBodyBytes, not the already-parsed/translated reqMap — can use the
// relaxed exploreReadWindow size (50) during Plan Mode too. An audit caught
// that both call sites hardcoded windowSize for "has modifying tools" at a
// flat 15-entry cap regardless of Plan Mode, while the trigger THRESHOLD
// checked in translator.go correctly relaxed to 40 for Plan Mode — a
// 15-entry window summed against a 40 threshold can never fire, silently
// disabling the guard for the exact case (Read/Grep-heavy Plan Mode
// investigation) it exists to protect.
func isPlanModeRequest(claudeBodyBytes []byte) bool {
	isPlanMode := false
	gjson.GetBytes(claudeBodyBytes, "tools").ForEach(func(_, t gjson.Result) bool {
		if strings.EqualFold(t.Get("name").String(), "exitplanmode") {
			isPlanMode = true
			return false
		}
		return true
	})
	return isPlanMode
}

// readLoopEquivalentToolName extracts the tool name from a "name:argsRaw"
// signature (sse.go's/the non-streaming path's toolCallSignatures) and
// reports whether this call counts as a Read for read-loop purposes —
// either literally the Read tool, or a Bash command paginating through a
// file the same way under a different name. Verified necessary live: an
// agent nudged off Read switched straight to `sed -n 'N,Mp' file` /
// `head`/`tail` chains in Bash and was then completely invisible to a
// counter that only ever matched the literal string "Read".
func readLoopEquivalentToolName(signature string) (name string, isReadEquivalent bool) {
	idx := strings.IndexByte(signature, ':')
	if idx < 0 {
		return signature, signature == "Read"
	}
	name = signature[:idx]
	if name == "Read" {
		return name, true
	}
	if name == "Bash" {
		command := gjson.Get(signature[idx+1:], "command").String()
		return name, isReadEquivalentBashCommand(command)
	}
	if name == "PowerShell" {
		command := gjson.Get(signature[idx+1:], "command").String()
		return name, isReadEquivalentPowerShellCommand(command)
	}
	return name, false
}

func readLoopLimits(agentKey string, hasModifyingTools bool) (windowSize, threshold int) {
	// ParseAgentInfo's fallback path (request-class "subagent" but no
	// x-claude-code-agent-type header) returns the bare string "subagent"
	// with no "sub:" prefix at all.
	if strings.HasPrefix(agentKey, "sub:") || agentKey == "subagent" || !hasModifyingTools {
		return exploreReadWindowSize, exploreReadWindowThreshold
	}
	return ReadWindowSize, ReadWindowThreshold
}

type LoopGuardManager struct {
	mu           sync.Mutex
	readWindows  map[string][]int     // agentKey -> recent per-turn Read-tool-call counts (sliding, max ReadWindowSize)
	toolCallHist map[string][]string  // agentKey -> list of turn call hashes
	lastSeen     map[string]time.Time // agentKey -> last activity, for staleness sweeps
	nudgeCount   map[string]int       // agentKey -> how many read-loop nudges fired since the last genuine Write/Edit; escalates wording instead of staying a one-time whisper
	editFailures map[string]int       // agentKey -> consecutive Edit tool_result is_error:true count
}

var globalLoopGuard = &LoopGuardManager{
	readWindows:  make(map[string][]int),
	toolCallHist: make(map[string][]string),
	lastSeen:     make(map[string]time.Time),
	nudgeCount:   make(map[string]int),
	editFailures: make(map[string]int),
}

// RecordEditResult tracks consecutive Edit tool_result failures per agent —
// the response-side madeProgress signal in server.go counts ANY Edit call
// as progress regardless of outcome (it can't know the outcome yet; the
// client reports is_error on the client's own NEXT request, in that
// request's tool_result, not in this gateway's response to the call that
// failed), so a model stuck retrying the exact same failing old_string
// indefinitely never tripped the read-loop guard. Returns the new
// consecutive-failure count (0 if this call was a success/reset it).
func (lg *LoopGuardManager) RecordEditResult(agentKey string, isError bool) int {
	lg.mu.Lock()
	defer lg.mu.Unlock()
	if isError {
		lg.editFailures[agentKey]++
		return lg.editFailures[agentKey]
	}
	lg.editFailures[agentKey] = 0
	return 0
}

func init() {
	go func() {
		ticker := time.NewTicker(loopGuardSweepEvery)
		defer ticker.Stop()
		for range ticker.C {
			globalLoopGuard.sweepStale()
		}
	}()
}

func (lg *LoopGuardManager) sweepStale() {
	lg.mu.Lock()
	defer lg.mu.Unlock()
	cutoff := time.Now().Add(-loopGuardStaleAfter)
	for k, t := range lg.lastSeen {
		if t.Before(cutoff) {
			delete(lg.lastSeen, k)
			delete(lg.readWindows, k)
			delete(lg.toolCallHist, k)
			delete(lg.nudgeCount, k)
			delete(lg.editFailures, k)
		}
	}
}

// touchLocked marks agentKey as active; caller must hold lg.mu.
func (lg *LoopGuardManager) touchLocked(agentKey string) {
	lg.lastSeen[agentKey] = time.Now()
}

// RecordReadTurn appends this turn's Read-tool-call count to agentKey's
// sliding window. madeProgress (a Write/Edit happened, or the turn carried a
// substantial amount of actual explanatory text/thinking — not just "let me
// check X") wipes the window: that's real forward motion, not skimming.
func (lg *LoopGuardManager) RecordReadTurn(agentKey string, readCount int, madeProgress bool, windowSize int) {
	if agentKey == "" || agentKey == "auxiliary" || strings.HasPrefix(agentKey, "auxiliary|") {
		return
	}
	lg.mu.Lock()
	defer lg.mu.Unlock()
	lg.touchLocked(agentKey)

	if madeProgress {
		delete(lg.readWindows, agentKey)
		delete(lg.nudgeCount, agentKey) // genuine progress earns a clean slate for the nudge wording too
		return
	}

	w := append(lg.readWindows[agentKey], readCount)
	if len(w) > windowSize {
		w = w[len(w)-windowSize:]
	}
	lg.readWindows[agentKey] = w
}

// CheckAndResetReadStreak reports the nudge's escalation level: 0 means no
// nudge fires. 1 is the first (soft) warning for this agent since its last
// genuine Write/Edit; 2+ means the soft nudge already fired and the agent
// slid straight back into another full read-loop window without ever
// writing/editing in between — confirmed real pattern (the model politely
// acknowledges the soft nudge, then immediately reads again), so the second
// firing gets much more directive wording instead of repeating the same
// whisper indefinitely.
// CheckAndResetReadStreak takes an explicit threshold instead of always
// using the package-level ReadWindowThreshold: a research subagent
// (Explore/Plan) is deliberately built to read 20-40+ files in a row with
// no Write/Edit at all — that's its entire job, not a loop — so the same
// 10-read threshold tuned for the main agent was cutting these subagents
// off mid-investigation and forcing them to return an incomplete answer.
// See readThresholdForAgent in translator.go, which picks the threshold
// passed in here based on the agent's role.
func (lg *LoopGuardManager) CheckAndResetReadStreak(agentKey string, threshold int) int {
	if agentKey == "" || agentKey == "auxiliary" || strings.HasPrefix(agentKey, "auxiliary|") {
		return 0
	}
	lg.mu.Lock()
	defer lg.mu.Unlock()

	total := 0
	for _, c := range lg.readWindows[agentKey] {
		total += c
	}
	if total >= threshold {
		delete(lg.readWindows, agentKey)
		lg.nudgeCount[agentKey]++
		return lg.nudgeCount[agentKey]
	}
	return 0
}

// RecordToolCalls records tool calls executed in this turn.
func (lg *LoopGuardManager) RecordToolCalls(agentKey string, calls []string, hasIgnoreTag bool) {
	if agentKey == "" || agentKey == "auxiliary" || strings.HasPrefix(agentKey, "auxiliary|") {
		return
	}
	lg.mu.Lock()
	defer lg.mu.Unlock()
	lg.touchLocked(agentKey)

	if hasIgnoreTag {
		lg.toolCallHist[agentKey] = nil
		return
	}

	if len(calls) == 0 {
		return
	}

	// Canonical sorted hash of calls in this turn
	sort.Strings(calls)
	h := sha256.New()
	for _, c := range calls {
		h.Write([]byte(c))
		h.Write([]byte{';'})
	}
	turnHash := hex.EncodeToString(h.Sum(nil))

	hist := lg.toolCallHist[agentKey]
	hist = append(hist, turnHash)
	if len(hist) > ToolCallHistMax {
		hist = hist[len(hist)-ToolCallHistMax:]
	}
	lg.toolCallHist[agentKey] = hist
}

func (lg *LoopGuardManager) CheckAndResetOscillation(agentKey string) bool {
	if agentKey == "" || agentKey == "auxiliary" || strings.HasPrefix(agentKey, "auxiliary|") {
		return false
	}
	lg.mu.Lock()
	defer lg.mu.Unlock()

	hist := lg.toolCallHist[agentKey]
	n := len(hist)

	// Detect period 1 (repeated x3), period 2 (A->B repeated x2), period 3 (A->B->C repeated x2)
	patterns := []struct {
		period int
		reps   int
	}{
		{1, 3},
		{2, 2},
		{3, 2},
	}

	for _, p := range patterns {
		need := p.period * p.reps
		if n < need {
			continue
		}
		window := hist[n-need:]
		pat := window[:p.period]

		// check non-degenerate pattern
		isDegenerate := false
		if p.period > 1 {
			unique := make(map[string]bool)
			for _, item := range pat {
				unique[item] = true
			}
			if len(unique) < p.period {
				isDegenerate = true
			}
		}
		if isDegenerate {
			continue
		}

		matched := true
		for i := 0; i < need; i++ {
			if window[i] != pat[i%p.period] {
				matched = false
				break
			}
		}

		if matched {
			lg.toolCallHist[agentKey] = nil
			return true
		}
	}

	return false
}

// InjectUserNudge injects a synthetic user message into the tail of the conversation
// without modifying system instructions, guaranteeing 100% byte invariance for the KV-cache prefix.
func InjectUserNudge(claudeReqJSON map[string]interface{}, text string) {
	msgsRaw, ok := claudeReqJSON["messages"]
	if !ok {
		return
	}
	var msgs []interface{}
	switch v := msgsRaw.(type) {
	case []interface{}:
		msgs = v
	case []map[string]interface{}:
		// An audit caught that this only ever accepted []interface{} —
		// SanitizeMessagesAndGraph's own normalization has previously
		// left "messages" as []map[string]interface{} at points in its
		// pipeline (see context_window.go's history of exactly this
		// mismatch), and a caller that still sees that shape here hit
		// this type assertion's failure path and silently did nothing,
		// with no error or log to notice it by. Maps are reference
		// types, so wrapping each one loses no ability to mutate it in
		// place below.
		msgs = make([]interface{}, len(v))
		for i, m := range v {
			msgs[i] = m
		}
	default:
		return
	}

	nudgeBlock := map[string]interface{}{"type": "text", "text": text}

	// If the conversation already ends on a "user" turn, append into it
	// instead of adding a second consecutive "user" message — Gemini
	// expects strict user/model alternation, and this guard fires exactly
	// when the model is misbehaving, which is the worst time to risk an
	// upstream 400 on top of it.
	if len(msgs) > 0 {
		if lastMsg, ok := msgs[len(msgs)-1].(map[string]interface{}); ok {
			if role, _ := lastMsg["role"].(string); role == "user" {
				switch rawContent := lastMsg["content"].(type) {
				case []map[string]interface{}:
					// An audit caught this switch only ever handled
					// []interface{} and string — a DIFFERENT gap than the
					// earlier fix above for msgsRaw's own outer type.
					// SanitizeMessagesAndGraph's normalization can leave a
					// message's "content" itself as []map[string]interface{},
					// and neither case here matched that shape, so this
					// whole block silently did nothing and fell through to
					// appending a brand new "user" message below —
					// producing two consecutive user turns, which
					// SanitizeMessagesAndGraph's merge then places a text
					// block as a sibling to a tool_result, which Gemini
					// rejects outright.
					content := make([]interface{}, len(rawContent))
					for i, m := range rawContent {
						content[i] = m
					}
					for _, item := range content {
						block, ok := item.(map[string]interface{})
						if !ok || block["type"] != "tool_result" {
							continue
						}
						switch c := block["content"].(type) {
						case string:
							block["content"] = c + "\n\n[" + text + "]"
						case []interface{}:
							block["content"] = append(c, map[string]interface{}{
								"type": "text",
								"text": "\n\n[" + text + "]",
							})
						default:
							block["content"] = "\n\n[" + text + "]"
						}
						lastMsg["content"] = content
						claudeReqJSON["messages"] = msgs
						return
					}
					lastMsg["content"] = append(content, nudgeBlock)
					claudeReqJSON["messages"] = msgs
					return
				case []interface{}:
					content := rawContent
					// Gemini requires a turn carrying a functionResponse to
					// contain ONLY functionResponse parts — a bare text block
					// appended alongside a tool_result here guarantees a 400
					// from Google. If this turn is a tool_result turn, fold
					// the nudge into the first tool_result's own content
					// instead of adding a sibling block.
					for _, item := range content {
						block, ok := item.(map[string]interface{})
						if !ok || block["type"] != "tool_result" {
							continue
						}
						switch c := block["content"].(type) {
						case string:
							block["content"] = c + "\n\n[" + text + "]"
						case []interface{}:
							block["content"] = append(c, map[string]interface{}{
								"type": "text",
								"text": "\n\n[" + text + "]",
							})
						default:
							block["content"] = "\n\n[" + text + "]"
						}
						claudeReqJSON["messages"] = msgs
						return
					}
					lastMsg["content"] = append(content, nudgeBlock)
					claudeReqJSON["messages"] = msgs
					return
				case string:
					lastMsg["content"] = []interface{}{
						map[string]interface{}{"type": "text", "text": rawContent},
						nudgeBlock,
					}
					claudeReqJSON["messages"] = msgs
					return
				}
			}
		}
	}

	nudgeMsg := map[string]interface{}{
		"role":    "user",
		"content": []map[string]interface{}{nudgeBlock},
	}

	claudeReqJSON["messages"] = append(msgs, nudgeMsg)
}
