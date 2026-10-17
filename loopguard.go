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

	// readToolGuidance is injected into every request's system prompt
	// (translator.go), silently and unconditionally — not a reactive
	// nudge after the fact, but an upfront instruction aimed at stopping
	// the common small-file case from ever needing one. Live traffic
	// showed the model defaulting to reading an entire file in a fixed
	// ~120-line page size regardless of the file's actual size, needing
	// 9 separate Read calls to get through a 1080-line file it could have
	// read in one.
	readToolGuidance = "Gateway notice (Read tool usage): if a file is small (roughly under 50KB, or a couple thousand lines), call Read once for the whole file instead of paging through it across multiple calls with offset/limit. Reserve offset/limit for genuinely large files. When you need to find something specific rather than read everything, prefer a targeted search (grep/rg via Bash) over sequential reads."

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
// raw file content" work the Read tool does — sed -n, head, tail, or cat
// — as opposed to a genuinely targeted search (grep/rg/awk), which is
// exactly what NudgeReadLoop itself tells the model to switch to and must
// never count against this same guard, or the nudge's own advice would
// trip it again immediately.
var readEquivalentBashPattern = regexp.MustCompile(`(?i)\b(sed\s+-n|head\s+-[nc]|tail\s+-[nc]|cat)\b`)

func isReadEquivalentBashCommand(command string) bool {
	lc := strings.ToLower(command)
	if strings.Contains(lc, "grep") || strings.Contains(lc, "awk") ||
		strings.Contains(lc, "rg ") || strings.HasPrefix(lc, "rg") {
		return false
	}
	return readEquivalentBashPattern.MatchString(command)
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
	msgs, ok := msgsRaw.([]interface{})
	if !ok {
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
				switch content := lastMsg["content"].(type) {
				case []interface{}:
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
						map[string]interface{}{"type": "text", "text": content},
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
