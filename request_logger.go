package main

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/tidwall/gjson"
	_ "modernc.org/sqlite"
)

type RequestLog struct {
	ID              uint64    `json:"id"`
	Timestamp       time.Time `json:"timestamp"`
	Key             string    `json:"key"`
	KeyMasked       string    `json:"key_masked"`
	Account         string    `json:"account"`
	SessionID       string    `json:"session_id"`
	AgentKey        string    `json:"agent_key"`  // "main", "sub:fork", "sub:explore", "auxiliary"
	AgentType       string    `json:"agent_type"` // "main", "subagent", "auxiliary"
	SubagentName    string    `json:"subagent_name,omitempty"`
	InstanceID      string    `json:"instance_id,omitempty"` // x-claude-code-prompt-id: stable per Task-tool invocation, distinguishes parallel same-named subagents
	Model           string    `json:"model"`
	TTFT            int64     `json:"ttft_ms"`
	TotalDur        int64     `json:"total_dur_ms"`
	InTokens        int64     `json:"in_tokens"`
	OutTokens       int64     `json:"out_tokens"`
	CacheReadTokens int64     `json:"cache_read_tokens"`
	CacheHitRate    float64   `json:"cache_hit_rate"`
	Tools           []string  `json:"tools"`
	WasPanicked     bool      `json:"was_panicked"`
	Shrunk          bool      `json:"shrunk"`
	Error           string    `json:"error,omitempty"`
	Status          int       `json:"status"`
}

type SQLiteLogStore struct {
	mu sync.RWMutex
	db *sql.DB
}

var globalLogStore = &SQLiteLogStore{}

func InitLogStore(dbPath string) error {
	if dbPath == "" {
		dbPath = "superrelay_logs.db"
	}
	db, err := sql.Open("sqlite", dbPath+"?_pragma=busy_timeout(5000)&_pragma=journal_mode(WAL)")
	if err != nil {
		return fmt.Errorf("open sqlite db: %w", err)
	}

	// Schema creation
	schema := `
	CREATE TABLE IF NOT EXISTS request_logs (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		timestamp DATETIME NOT NULL,
		client_key TEXT NOT NULL,
		key_masked TEXT NOT NULL,
		account TEXT NOT NULL,
		session_id TEXT NOT NULL,
		agent_key TEXT NOT NULL DEFAULT 'main',
		agent_type TEXT NOT NULL DEFAULT 'main',
		subagent_name TEXT NOT NULL DEFAULT '',
		instance_id TEXT NOT NULL DEFAULT '',
		model TEXT NOT NULL,
		ttft_ms INTEGER NOT NULL DEFAULT 0,
		total_dur_ms INTEGER NOT NULL DEFAULT 0,
		in_tokens INTEGER NOT NULL DEFAULT 0,
		out_tokens INTEGER NOT NULL DEFAULT 0,
		cache_read_tokens INTEGER NOT NULL DEFAULT 0,
		cache_hit_rate REAL NOT NULL DEFAULT 0,
		tools_json TEXT NOT NULL DEFAULT '[]',
		was_panicked INTEGER NOT NULL DEFAULT 0,
		shrunk INTEGER NOT NULL DEFAULT 0,
		error TEXT NOT NULL DEFAULT '',
		status INTEGER NOT NULL DEFAULT 200
	);
	CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON request_logs(timestamp DESC);
	CREATE INDEX IF NOT EXISTS idx_logs_session ON request_logs(session_id);
	CREATE INDEX IF NOT EXISTS idx_logs_account ON request_logs(account);
	CREATE INDEX IF NOT EXISTS idx_logs_client_key ON request_logs(client_key);
	CREATE INDEX IF NOT EXISTS idx_logs_agent ON request_logs(agent_key);
	`
	if _, err := db.Exec(schema); err != nil {
		return fmt.Errorf("exec schema: %w", err)
	}
	// Migration for DBs created before instance_id existed; sqlite has no
	// "ADD COLUMN IF NOT EXISTS", so just swallow the "duplicate column" error.
	_, _ = db.Exec(`ALTER TABLE request_logs ADD COLUMN instance_id TEXT NOT NULL DEFAULT ''`)

	globalLogStore.db = db

	// Clean up records older than 24h at startup and periodically
	go func() {
		cleanupOldLogs(db)
		ticker := time.NewTicker(1 * time.Hour)
		defer ticker.Stop()
		for range ticker.C {
			cleanupOldLogs(db)
		}
	}()

	// WAL mode never shrinks its -wal file on its own without an explicit
	// checkpoint; left unattended it grows unbounded (seen in practice: 4MB+
	// WAL against a 220KB main db file) since nothing else in this gateway
	// ever asked SQLite to fold it back in.
	go func() {
		ticker := time.NewTicker(15 * time.Minute)
		defer ticker.Stop()
		for range ticker.C {
			_, _ = db.Exec("PRAGMA wal_checkpoint(TRUNCATE)")
		}
	}()

	return nil
}

func cleanupOldLogs(db *sql.DB) {
	cutoff := time.Now().Add(-24 * time.Hour)
	_, _ = db.Exec("DELETE FROM request_logs WHERE timestamp < ?", cutoff)
}

// Close checkpoints the WAL back into the main db file and closes the
// connection. Called on graceful shutdown so a killed process doesn't leave
// an uncheckpointed -wal file (or, worse, miss flushing the last batch of
// in-flight writes) behind.
func (s *SQLiteLogStore) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.db == nil {
		return
	}
	_, _ = s.db.Exec("PRAGMA wal_checkpoint(TRUNCATE)")
	_ = s.db.Close()
}

func MaskKey(key string) string {
	if len(key) <= 14 {
		return key
	}
	return key[:8] + "..." + key[len(key)-4:]
}

func parseSQLiteTime(s string) time.Time {
	// The driver stores bound time.Time values using Go's default
	// time.Time.String() layout ("2006-01-02 15:04:05.999999999 -0700 MST"),
	// with a " m=+<seconds>" monotonic-clock suffix appended whenever the
	// value carries a monotonic reading (true of every time.Now()-derived
	// timestamp this codebase logs). A direct-to-time.Time db.Scan handles
	// this internally via the driver, but MIN()/MAX() aggregates force the
	// result through a plain string first — and that monotonic suffix
	// isn't a layout time.Parse understands, so every one of these used to
	// fail silently and fall through to the time.Now() fallback below
	// (confirmed: every session's start_time/last_time was just "now").
	if idx := strings.Index(s, " m="); idx >= 0 {
		s = s[:idx]
	}

	formats := []string{
		"2006-01-02 15:04:05.999999999 -0700 MST",
		time.RFC3339Nano,
		time.RFC3339,
		"2006-01-02 15:04:05.999999999-07:00",
		"2006-01-02 15:04:05-07:00",
		"2006-01-02 15:04:05",
		"2006-01-02T15:04:05",
	}
	for _, f := range formats {
		if t, err := time.Parse(f, s); err == nil {
			return t
		}
	}
	return time.Now()
}

// ParseAgentInfo also returns instanceID, used to tell apart N concurrent
// same-named subagents (e.g. 3 simultaneous Explore calls, which would
// otherwise all collapse into one indistinguishable "sub:Explore" branch).
//
// An earlier version of this comment claimed x-claude-code-prompt-id was
// the stable per-instance key and x-claude-code-agent-id was not. That was
// backwards, confirmed by two independent live captures (this gateway's
// own local traffic, and genuine claude-cli traffic against real
// api.anthropic.com via mitmdump): when the orchestrator spawns two
// parallel "general-purpose" subagents in one "Agent" tool_use batch, BOTH
// share the exact same Prompt-Id for their entire run, while each gets its
// own Agent-Id that stays constant across all of its own turns. Prompt-Id
// is effectively the *batch* id (shared by every subagent launched
// together); Agent-Id is the *instance* id. Only subagents carry
// Agent-Id at all (the main/orchestrator thread never does, per
// claude-cli's own Kp(agentType==="main") check), which is also why it's
// never used as a fallback for "main"/"auxiliary" below.
func ParseAgentInfo(r *http.Request) (agentKey, agentType, subagentName, instanceID string) {
	reqClass := r.Header.Get("x-claude-code-request-class")
	rawAgentType := r.Header.Get("x-claude-code-agent-type")
	promptID := r.Header.Get("x-claude-code-prompt-id")
	agentID := r.Header.Get("x-claude-code-agent-id")

	if reqClass == "auxiliary" {
		return "auxiliary", "auxiliary", "", promptID
	}
	if rawAgentType != "" {
		if agentID != "" {
			return "sub:" + rawAgentType, "subagent", rawAgentType, agentID
		}
		return "sub:" + rawAgentType, "subagent", rawAgentType, promptID
	}
	if reqClass == "subagent" {
		if agentID != "" {
			return "subagent", "subagent", "subagent", agentID
		}
		return "subagent", "subagent", "subagent", promptID
	}
	return "main", "main", "", promptID
}

func DeriveSessionID(r *http.Request, bodyBytes []byte, geminiBody []byte) string {
	if s := r.Header.Get("x-session-id"); s != "" {
		return s
	}
	if s := r.Header.Get("x-claude-code-session-id"); s != "" {
		return s
	}
	if s := gjson.GetBytes(bodyBytes, "metadata.session_id").String(); s != "" {
		return s
	}

	msgs := gjson.GetBytes(bodyBytes, "messages")
	if msgs.IsArray() {
		for _, m := range msgs.Array() {
			if m.Get("role").String() == "user" {
				txt := m.Get("content.0.text").String()
				if txt == "" {
					txt = m.Get("content").String()
				}
				if txt != "" {
					h := sha256.Sum256([]byte(txt))
					return "sess-" + hex.EncodeToString(h[:4])
				}
			}
		}
	}

	rawSess := generateStableSessionID(geminiBody, 0, "")
	rawClean := strings.TrimPrefix(rawSess, "-")
	if len(rawClean) > 8 {
		rawClean = rawClean[:8]
	}
	return "sess-" + rawClean
}

func (s *SQLiteLogStore) Add(entry RequestLog) {
	if s.db == nil {
		return
	}
	if entry.Timestamp.IsZero() {
		entry.Timestamp = time.Now()
	}
	if entry.Tools == nil {
		entry.Tools = []string{}
	}
	if entry.AgentKey == "" {
		entry.AgentKey = "main"
	}
	if entry.AgentType == "" {
		entry.AgentType = "main"
	}
	toolsBytes, _ := json.Marshal(entry.Tools)

	wasPanickedInt := 0
	if entry.WasPanicked {
		wasPanickedInt = 1
	}
	shrunkInt := 0
	if entry.Shrunk {
		shrunkInt = 1
	}

	query := `
	INSERT INTO request_logs (
		timestamp, client_key, key_masked, account, session_id,
		agent_key, agent_type, subagent_name, instance_id, model,
		ttft_ms, total_dur_ms, in_tokens, out_tokens, cache_read_tokens,
		cache_hit_rate, tools_json, was_panicked, shrunk, error, status
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
	`
	// Only the masked form of the key is ever persisted to disk, in both
	// columns — storing the raw tenant API key in a plaintext SQLite file
	// (and returning it via /api/logs) was an avoidable sensitive-data-at-
	// rest exposure the dashboard password doesn't fully excuse.
	_, _ = s.db.Exec(
		query,
		entry.Timestamp, entry.KeyMasked, entry.KeyMasked, entry.Account, entry.SessionID,
		entry.AgentKey, entry.AgentType, entry.SubagentName, entry.InstanceID, entry.Model,
		entry.TTFT, entry.TotalDur, entry.InTokens, entry.OutTokens, entry.CacheReadTokens,
		entry.CacheHitRate, string(toolsBytes), wasPanickedInt, shrunkInt, entry.Error, entry.Status,
	)

	// Broadcast in real-time over WebSocket to dashboard clients
	go func(e RequestLog) {
		globalWSHub.BroadcastLog(&e)
		globalWSHub.BroadcastSessionsUpdate()
	}(entry)
}

func (s *SQLiteLogStore) Clear() {
	if s.db == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	_, _ = s.db.Exec("DELETE FROM request_logs")
}

type LogStats struct {
	TotalRequests   int     `json:"total_requests"`
	AvgCacheHitRate float64 `json:"avg_cache_hit_rate"`
	AvgTTFTMs       int64   `json:"avg_ttft_ms"`
	ActiveSessions  int     `json:"active_sessions"`
	ErrorCount      int     `json:"error_count"`
	PanicCount      int     `json:"panic_count"`
	TotalAccounts   int     `json:"total_accounts"`
	ActiveKeys      int     `json:"active_keys"`
}

type SessionSummary struct {
	SessionID      string    `json:"session_id"`
	Key            string    `json:"key"`
	KeyMasked      string    `json:"key_masked"`
	Account        string    `json:"account"`
	StartTime      time.Time `json:"start_time"`
	LastTime       time.Time `json:"last_time"`
	TotalRequests  int       `json:"total_requests"`
	TotalInTokens  int64     `json:"total_in_tokens"`
	TotalOutTokens int64     `json:"total_out_tokens"`
	CacheHitRate   float64   `json:"cache_hit_rate"`
	MainRequests   int       `json:"main_requests"`
	SubRequests    int       `json:"sub_requests"`
	SubagentNames  []string  `json:"subagent_names"`
	HasErrors      bool      `json:"has_errors"`
	HasPanic       bool      `json:"has_panic"`
}

func (s *SQLiteLogStore) QuerySessions(limit int) []SessionSummary {
	if s.db == nil {
		return []SessionSummary{}
	}
	s.mu.RLock()
	defer s.mu.RUnlock()

	if limit <= 0 || limit > 100 {
		limit = 50
	}

	query := `
	SELECT
		session_id,
		client_key,
		key_masked,
		account,
		MIN(timestamp) as start_time,
		MAX(timestamp) as last_time,
		COUNT(*) as total_requests,
		SUM(in_tokens) as total_in,
		SUM(out_tokens) as total_out,
		COALESCE(AVG(CASE WHEN in_tokens > 0 THEN cache_hit_rate END), 0) as avg_hit,
		SUM(CASE WHEN agent_type = 'main' THEN 1 ELSE 0 END) as main_count,
		SUM(CASE WHEN agent_type = 'subagent' THEN 1 ELSE 0 END) as sub_count,
		SUM(CASE WHEN status >= 400 OR error != '' THEN 1 ELSE 0 END) as err_count,
		SUM(CASE WHEN was_panicked = 1 THEN 1 ELSE 0 END) as panic_count,
		GROUP_CONCAT(DISTINCT CASE WHEN subagent_name != '' THEN subagent_name END) as subagent_names
	FROM request_logs
	WHERE session_id != '-' AND session_id != ''
	GROUP BY session_id
	ORDER BY last_time DESC
	LIMIT ?;
	`

	rows, err := s.db.Query(query, limit)
	if err != nil {
		return []SessionSummary{}
	}
	defer rows.Close()

	var list []SessionSummary = make([]SessionSummary, 0)
	for rows.Next() {
		var sm SessionSummary
		var totalIn, totalOut sql.NullInt64
		var avgHit sql.NullFloat64
		var mainCount, subCount, errCount, panicCount sql.NullInt64
		var startTimeStr, lastTimeStr string
		var subagentNamesRaw sql.NullString

		err := rows.Scan(
			&sm.SessionID, &sm.Key, &sm.KeyMasked, &sm.Account,
			&startTimeStr, &lastTimeStr,
			&sm.TotalRequests, &totalIn, &totalOut,
			&avgHit, &mainCount, &subCount,
			&errCount, &panicCount, &subagentNamesRaw,
		)
		if err != nil {
			continue
		}
		sm.StartTime = parseSQLiteTime(startTimeStr)
		sm.LastTime = parseSQLiteTime(lastTimeStr)
		if totalIn.Valid {
			sm.TotalInTokens = totalIn.Int64
		}
		if totalOut.Valid {
			sm.TotalOutTokens = totalOut.Int64
		}
		if avgHit.Valid {
			sm.CacheHitRate = avgHit.Float64
		}
		if mainCount.Valid {
			sm.MainRequests = int(mainCount.Int64)
		}
		if subCount.Valid {
			sm.SubRequests = int(subCount.Int64)
		}
		if errCount.Valid {
			sm.HasErrors = (errCount.Int64 > 0)
		}
		if panicCount.Valid {
			sm.HasPanic = (panicCount.Int64 > 0)
		}

		if subagentNamesRaw.Valid && subagentNamesRaw.String != "" {
			sm.SubagentNames = strings.Split(subagentNamesRaw.String, ",")
		}
		if sm.SubagentNames == nil {
			sm.SubagentNames = []string{}
		}

		list = append(list, sm)
	}

	return list
}

// GetSessionDetail returns the full picture for exactly one session: its
// aggregate summary (same shape as QuerySessions' rows) plus every request
// logged under it, in chronological (ascending) order — unlike Query(),
// which caps at 500 and sorts newest-first. Built for MCP-side verification
// of a specific session's behavior (model routing, cache hits, panics,
// errors), where silently truncating the timeline would hide exactly the
// failure being checked for.
func (s *SQLiteLogStore) GetSessionDetail(sessionID string) (*SessionSummary, []RequestLog) {
	if s.db == nil || sessionID == "" {
		return nil, nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()

	summaryQuery := `
	SELECT
		session_id,
		client_key,
		key_masked,
		account,
		MIN(timestamp) as start_time,
		MAX(timestamp) as last_time,
		COUNT(*) as total_requests,
		SUM(in_tokens) as total_in,
		SUM(out_tokens) as total_out,
		COALESCE(AVG(CASE WHEN in_tokens > 0 THEN cache_hit_rate END), 0) as avg_hit,
		SUM(CASE WHEN agent_type = 'main' THEN 1 ELSE 0 END) as main_count,
		SUM(CASE WHEN agent_type = 'subagent' THEN 1 ELSE 0 END) as sub_count,
		SUM(CASE WHEN status >= 400 OR error != '' THEN 1 ELSE 0 END) as err_count,
		SUM(CASE WHEN was_panicked = 1 THEN 1 ELSE 0 END) as panic_count
	FROM request_logs
	WHERE session_id = ?
	GROUP BY session_id;
	`

	var sm SessionSummary
	var totalIn, totalOut sql.NullInt64
	var avgHit sql.NullFloat64
	var mainCount, subCount, errCount, panicCount sql.NullInt64
	var startTimeStr, lastTimeStr string

	row := s.db.QueryRow(summaryQuery, sessionID)
	err := row.Scan(
		&sm.SessionID, &sm.Key, &sm.KeyMasked, &sm.Account,
		&startTimeStr, &lastTimeStr,
		&sm.TotalRequests, &totalIn, &totalOut,
		&avgHit, &mainCount, &subCount,
		&errCount, &panicCount,
	)
	if err != nil {
		return nil, nil // no rows for this session_id
	}
	sm.StartTime = parseSQLiteTime(startTimeStr)
	sm.LastTime = parseSQLiteTime(lastTimeStr)
	if totalIn.Valid {
		sm.TotalInTokens = totalIn.Int64
	}
	if totalOut.Valid {
		sm.TotalOutTokens = totalOut.Int64
	}
	if avgHit.Valid {
		sm.CacheHitRate = avgHit.Float64
	}
	if mainCount.Valid {
		sm.MainRequests = int(mainCount.Int64)
	}
	if subCount.Valid {
		sm.SubRequests = int(subCount.Int64)
	}
	if errCount.Valid {
		sm.HasErrors = errCount.Int64 > 0
	}
	if panicCount.Valid {
		sm.HasPanic = panicCount.Int64 > 0
	}

	subRows, err := s.db.Query(`SELECT DISTINCT subagent_name FROM request_logs WHERE session_id = ? AND subagent_name != ''`, sessionID)
	if err == nil {
		for subRows.Next() {
			var subName string
			if err := subRows.Scan(&subName); err == nil && subName != "" {
				sm.SubagentNames = append(sm.SubagentNames, subName)
			}
		}
		subRows.Close()
	}
	if sm.SubagentNames == nil {
		sm.SubagentNames = []string{}
	}

	detailQuery := `
	SELECT
		id, timestamp, client_key, key_masked, account, session_id,
		agent_key, agent_type, subagent_name, instance_id, model,
		ttft_ms, total_dur_ms, in_tokens, out_tokens, cache_read_tokens,
		cache_hit_rate, tools_json, was_panicked, shrunk, error, status
	FROM request_logs
	WHERE session_id = ?
	ORDER BY timestamp ASC
	LIMIT 2000;
	`
	rows, err := s.db.Query(detailQuery, sessionID)
	if err != nil {
		return &sm, []RequestLog{}
	}
	defer rows.Close()

	requests := make([]RequestLog, 0)
	for rows.Next() {
		var item RequestLog
		var toolsJSON string
		var wasPanickedInt, shrunkInt int
		err := rows.Scan(
			&item.ID, &item.Timestamp, &item.Key, &item.KeyMasked, &item.Account, &item.SessionID,
			&item.AgentKey, &item.AgentType, &item.SubagentName, &item.InstanceID, &item.Model,
			&item.TTFT, &item.TotalDur, &item.InTokens, &item.OutTokens, &item.CacheReadTokens,
			&item.CacheHitRate, &toolsJSON, &wasPanickedInt, &shrunkInt, &item.Error, &item.Status,
		)
		if err != nil {
			continue
		}
		item.WasPanicked = wasPanickedInt == 1
		item.Shrunk = shrunkInt == 1
		_ = json.Unmarshal([]byte(toolsJSON), &item.Tools)
		if item.Tools == nil {
			item.Tools = []string{}
		}
		requests = append(requests, item)
	}

	return &sm, requests
}

func (s *SQLiteLogStore) Query(keyFilter, accountFilter, sessionFilter, agentFilter string, onlyErrors bool, limit int) ([]RequestLog, LogStats) {
	if s.db == nil {
		return []RequestLog{}, LogStats{}
	}
	s.mu.RLock()
	defer s.mu.RUnlock()

	if limit <= 0 || limit > 500 {
		limit = 100
	}

	// Calculate overall 24h stats
	statsQuery := `
	SELECT
		COUNT(*),
		COALESCE(AVG(CASE WHEN in_tokens > 0 THEN cache_hit_rate END), 0),
		COALESCE(AVG(CASE WHEN ttft_ms > 0 THEN ttft_ms END), 0),
		COUNT(DISTINCT CASE WHEN session_id != '-' AND session_id != '' THEN session_id END),
		SUM(CASE WHEN status >= 400 OR error != '' THEN 1 ELSE 0 END),
		SUM(CASE WHEN was_panicked = 1 THEN 1 ELSE 0 END)
	FROM request_logs;
	`
	var stats LogStats
	row := s.db.QueryRow(statsQuery)
	var avgCacheHit, avgTTFT float64
	var errCount, panicCount sql.NullInt64
	_ = row.Scan(&stats.TotalRequests, &avgCacheHit, &avgTTFT, &stats.ActiveSessions, &errCount, &panicCount)
	stats.AvgCacheHitRate = avgCacheHit
	stats.AvgTTFTMs = int64(avgTTFT)
	if errCount.Valid {
		stats.ErrorCount = int(errCount.Int64)
	}
	if panicCount.Valid {
		stats.PanicCount = int(panicCount.Int64)
	}

	// Build WHERE clause for filtered list
	var conditions []string
	var args []interface{}

	if keyFilter != "" {
		conditions = append(conditions, "(client_key LIKE ? OR key_masked LIKE ?)")
		args = append(args, "%"+keyFilter+"%", "%"+keyFilter+"%")
	}
	if accountFilter != "" {
		conditions = append(conditions, "account = ?")
		args = append(args, accountFilter)
	}
	if sessionFilter != "" {
		conditions = append(conditions, "session_id LIKE ?")
		args = append(args, "%"+sessionFilter+"%")
	}
	if agentFilter != "" {
		conditions = append(conditions, "(agent_key = ? OR agent_type = ? OR subagent_name = ?)")
		args = append(args, agentFilter, agentFilter, agentFilter)
	}
	if onlyErrors {
		conditions = append(conditions, "(status >= 400 OR error != '' OR was_panicked = 1)")
	}

	whereClause := ""
	if len(conditions) > 0 {
		whereClause = "WHERE " + strings.Join(conditions, " AND ")
	}

	query := fmt.Sprintf(`
	SELECT
		id, timestamp, client_key, key_masked, account, session_id,
		agent_key, agent_type, subagent_name, instance_id, model,
		ttft_ms, total_dur_ms, in_tokens, out_tokens, cache_read_tokens,
		cache_hit_rate, tools_json, was_panicked, shrunk, error, status
	FROM request_logs
	%s
	ORDER BY timestamp DESC
	LIMIT ?;
	`, whereClause)
	args = append(args, limit)

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return []RequestLog{}, stats
	}
	defer rows.Close()

	var logs []RequestLog
	for rows.Next() {
		var item RequestLog
		var toolsJSON string
		var wasPanickedInt, shrunkInt int
		err := rows.Scan(
			&item.ID, &item.Timestamp, &item.Key, &item.KeyMasked, &item.Account, &item.SessionID,
			&item.AgentKey, &item.AgentType, &item.SubagentName, &item.InstanceID, &item.Model,
			&item.TTFT, &item.TotalDur, &item.InTokens, &item.OutTokens, &item.CacheReadTokens,
			&item.CacheHitRate, &toolsJSON, &wasPanickedInt, &shrunkInt, &item.Error, &item.Status,
		)
		if err != nil {
			continue
		}
		item.WasPanicked = (wasPanickedInt == 1)
		item.Shrunk = (shrunkInt == 1)
		_ = json.Unmarshal([]byte(toolsJSON), &item.Tools)
		if item.Tools == nil {
			item.Tools = []string{}
		}
		logs = append(logs, item)
	}

	return logs, stats
}
