package main

import (
	"container/list"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/tidwall/gjson"
)

const (
	// sessionAffinityTTL outlives the longest prompt cache TTL (1h), so a
	// binding never lapses while its account can still hold the cache.
	sessionAffinityTTL        = 1 * time.Hour
	maxSessionAffinityEntries = 5000
)

// affinityLRUEntry is the value stored in each affinityLRU list.Element.
// Kept as a pointer-sized struct in the list itself (not a copy in the map)
// so lookups, refreshes and moves are all O(1) without re-allocating.
type affinityLRUEntry struct {
	key       string
	email     string
	expiresAt time.Time
}

type AccountSelector struct {
	mu        sync.Mutex
	stickyMap map[string]string // apiKey -> assigned email (fallback)

	// affinityLRU/affinityIdx together replace a plain
	// map[string]SessionAffinityEntry. A bare Go map has no iteration
	// order guarantee, so evicting "the oldest" out of it at capacity was
	// actually evicting an arbitrary entry — on a long-lived server this
	// could silently kick an ACTIVE conversation's account binding and
	// force an avoidable Google-side prompt-cache reset. list.List gives a
	// real most-recently-used-at-front order: MoveToFront on every hit,
	// evict from Back() when full, so eviction is deterministic and always
	// picks the conversation that's gone longest without being touched.
	affinityLRU   *list.List               // front = most recently used
	affinityIdx   map[string]*list.Element // conversationAnchor/sessionID -> its node in affinityLRU
	roundRobinIdx int
}

var globalSelector = &AccountSelector{
	stickyMap:   make(map[string]string),
	affinityLRU: list.New(),
	affinityIdx: make(map[string]*list.Element),
}

// DeriveConversationAnchor creates a stable identifier for a multi-turn conversation.
// Every turn of the same conversation shares the same anchor (first user message + system prefix),
// allowing all turns to hit the exact same Google account for 100% prompt caching.
//
// headerSessionID is Claude Code's own terminal-session id
// (x-claude-code-session-id / x-session-id), which — unlike the first user
// message — never changes for the life of the conversation. Without it, a
// long conversation that crosses TruncateConversationToFit's drop threshold
// gets its oldest messages replaced by a synthetic "[Gateway notice: ...]"
// message; on the very next turn THAT notice becomes the new "first user
// message", the hash changes, and the conversation gets silently rebound to
// a different Google account with none of the prior turns' prefix cache.
// instanceID, when non-empty, is a subagent's own stable per-instance key
// (see ParseAgentInfo) — headerSessionID alone is shared by the main thread
// and EVERY subagent in the conversation, so without qualifying by it here,
// Session Affinity pinned 5-10 parallel subagents to the exact same Google
// account (they share no content worth caching together anyway), piling
// all of their load onto one account's quota and triggering an immediate
// 429 instead of spreading across the pool the way the other per-instance
// keys (antigravitySessionKey, cwdKey, loopKey) already do.
func DeriveConversationAnchor(bodyBytes []byte, headerSessionID string, instanceID string) string {
	if headerSessionID != "" {
		if instanceID != "" {
			return "sid:" + headerSessionID + ":" + instanceID
		}
		return "sid:" + headerSessionID
	}

	root := gjson.ParseBytes(bodyBytes)

	// If client explicitly sent a session id, use it
	if sid := root.Get("metadata.session_id").String(); sid != "" {
		return "sid:" + sid
	}

	// Find the first user message — skipping this gateway's own injected
	// truncation notices, so the anchor stays on the oldest surviving real
	// message instead of flipping to the notice text itself.
	firstUserMsg := ""
	msgs := root.Get("messages")
	if msgs.IsArray() {
		for _, m := range msgs.Array() {
			if m.Get("role").String() == "user" {
				c := m.Get("content")
				candidate := ""
				if c.Type == gjson.String {
					candidate = c.String()
				} else if c.IsArray() {
					for _, part := range c.Array() {
						if part.Get("type").String() == "text" {
							candidate = part.Get("text").String()
							break
						}
					}
				}
				if strings.HasPrefix(strings.TrimSpace(candidate), "[Gateway notice:") {
					continue
				}
				if candidate != "" {
					firstUserMsg = candidate
					break
				}
			}
		}
	}

	if firstUserMsg == "" {
		return ""
	}

	// Hash first user message (up to first 500 chars to avoid oversized inputs).
	// NOTE: We deliberately DO NOT include 'model' in the conversation hash!
	// Claude Code frequently sends auxiliary requests (summaries, titles) using sonnet/haiku
	// while main turns use opus/flash. Excluding model ensures 100% stable Session Affinity
	// and preserves prompt cache across all turns of the same conversation.
	cleanFirst := strings.TrimSpace(firstUserMsg)
	if len(cleanFirst) > 500 {
		cleanFirst = cleanFirst[:500]
	}

	h := sha256.Sum256([]byte(cleanFirst))
	return "conv:" + hex.EncodeToString(h[:12])
}

// SelectAccount chooses an account for a request based on tenant strategy,
// prioritizing Session Affinity by Conversation Anchor for maximum prompt caching.
func (as *AccountSelector) SelectAccount(apiKey string, conversationKey string, tenant TenantConfig) (*AccountRecord, error) {
	return as.SelectAccountExcluding(apiKey, conversationKey, tenant, nil)
}

// SelectAccountExcluding chooses an account while skipping accounts in the excluded map (useful for Failover).
func (as *AccountSelector) SelectAccountExcluding(apiKey string, conversationKey string, tenant TenantConfig, excluded map[string]bool) (*AccountRecord, error) {
	// Deep-copies every account's Quotas slice — fetched before taking
	// as.mu (it only touches AccountManager's own lock, nothing in as)
	// so this snapshot cost isn't paid while holding the selector's lock
	// and serializing every OTHER concurrent request's own account pick
	// behind it.
	accounts := globalAccountManager.GetAccounts()

	as.mu.Lock()
	defer as.mu.Unlock()

	if len(accounts) == 0 {
		return nil, fmt.Errorf("no accounts available in pool")
	}

	// Filter out excluded accounts
	isExcluded := func(acc *AccountRecord) bool {
		return acc.Disabled || (excluded != nil && excluded[acc.Email])
	}

	// 1. Explicitly pinned in config
	if tenant.StickyAccount != "" {
		for _, acc := range accounts {
			if acc.Email == tenant.StickyAccount && !isExcluded(acc) {
				return acc, nil
			}
		}
	}

	now := time.Now()

	// 2. Session Affinity by Conversation Anchor (Highest Priority for prompt cache)
	if conversationKey != "" {
		if el, ok := as.affinityIdx[conversationKey]; ok {
			entry := el.Value.(*affinityLRUEntry)
			if now.Before(entry.expiresAt) {
				for _, acc := range accounts {
					if acc.Email == entry.email && !isExcluded(acc) && hasRemainingQuota(acc) {
						// Refresh lease and mark as most-recently-used.
						entry.expiresAt = now.Add(sessionAffinityTTL)
						as.affinityLRU.MoveToFront(el)
						return acc, nil
					}
				}
			} else {
				as.affinityLRU.Remove(el)
				delete(as.affinityIdx, conversationKey)
			}
		}
	}

	// 3. Fallback: Sticky per API Key — main thread / non-subagent requests
	// only. A subagent's conversationKey is "sid:<session>:<instanceID>"
	// (see DeriveConversationAnchor); on its very FIRST turn, step 2 above
	// always misses (nothing has bound that exact composite key yet), so
	// without this guard every subagent fell straight through to the same
	// per-apiKey sticky account every OTHER request already uses — there's
	// only one stickyMap entry per apiKey, shared by the main thread and
	// every subagent, so this collapsed 5-10 parallel subagents onto the
	// exact same Google account on their first request and then PERSISTED
	// that via RememberAffinity on success, undoing the whole point of
	// qualifying conversationKey by instanceID in the first place.
	isSubagentKey := strings.Count(conversationKey, ":") >= 2
	if !isSubagentKey && (tenant.Strategy == "sticky" || tenant.Strategy == "") && apiKey != "" {
		if assignedEmail, ok := as.stickyMap[apiKey]; ok {
			for _, acc := range accounts {
				if acc.Email == assignedEmail && !isExcluded(acc) && hasRemainingQuota(acc) {
					// An audit caught that binding session affinity HERE —
					// at selection time, before any network attempt with
					// this account has even started — meant a retry
					// (SelectAccountExcluding called again with this
					// account now excluded after a transient
					// timeout/429) immediately overwrote the conversation's
					// affinity to whichever account the retry picked next,
					// even if that pick never ends up succeeding either.
					// RememberAffinity already does this correctly, gated
					// on an actual confirmed HTTP 200 (server.go), so the
					// speculative bind here only ever destroyed the
					// previous account's warm KV-cache for no benefit.
					return acc, nil
				}
			}
		}
	}

	// 4. For subagents on their first turn, or if tenant strategy is round_robin,
	// distribute across the pool using round-robin among healthy accounts rather than
	// greedily picking the single account with highest quota fraction (which sends 3-5 concurrent
	// subagents to the exact same account simultaneously, triggering 429s).
	if isSubagentKey || tenant.Strategy == "round_robin" {
		valid := make([]*AccountRecord, 0)
		for _, acc := range accounts {
			if !isExcluded(acc) && hasRemainingQuota(acc) {
				valid = append(valid, acc)
			}
		}
		if len(valid) > 0 {
			as.roundRobinIdx = (as.roundRobinIdx + 1) % len(valid)
			chosen := valid[as.roundRobinIdx]
			if !isSubagentKey && apiKey != "" {
				as.stickyMap[apiKey] = chosen.Email
			}
			// Speculative affinity bind removed here too — see the sticky
			// branch above for why; RememberAffinity (server.go, gated on
			// a confirmed 200) is now the only place affinity gets set.
			return chosen, nil
		}
	}

	// 4b. Select the best available account based on remaining quota
	var bestAcc *AccountRecord
	var maxQuota float64 = -1.0

	for _, acc := range accounts {
		if isExcluded(acc) {
			continue
		}
		q := getEffectiveQuotaFraction(acc)
		if q > maxQuota {
			maxQuota = q
			bestAcc = acc
		}
	}

	// maxQuota starts at -1.0, so a pool where every account is sitting at
	// 0% remaining quota still satisfies q(0.0) > maxQuota(-1.0) on the
	// very first account iterated — bestAcc was never nil in that case,
	// so step 5's honest round-robin below was dead code: every request
	// piled onto whichever single exhausted account iteration order
	// happened to try first, instead of spreading load across the pool
	// while everyone waits out the same cooldown.
	if bestAcc != nil && maxQuota > 0 {
		if !isSubagentKey && apiKey != "" {
			as.stickyMap[apiKey] = bestAcc.Email
		}
		// Speculative affinity bind removed here too — see the sticky
		// branch above for why.
		return bestAcc, nil
	}

	// 5. Fallback: Round-Robin across any non-excluded accounts not
	// currently in cooldown.
	//
	// An audit caught that this filtered only on isExcluded (accounts
	// already tried THIS request) and never checked CooldownUntil at
	// all — reaching this step means step 4b's maxQuota>0 check already
	// failed, i.e. every account's effective quota fraction is at or near
	// zero, overwhelmingly because they're all in cooldown. Without this
	// check, every one of the caller's retry attempts picked a
	// guaranteed-to-fail cooling-down account and burned a real network
	// round-trip on it, instead of recognizing the pool is genuinely
	// exhausted and failing fast with 429.
	valid := make([]*AccountRecord, 0)
	for _, acc := range accounts {
		if !isExcluded(acc) && (acc.CooldownUntil.IsZero() || !now.Before(acc.CooldownUntil)) {
			valid = append(valid, acc)
		}
	}

	if len(valid) == 0 {
		return nil, fmt.Errorf("all accounts are disabled, cooling down or excluded")
	}

	as.roundRobinIdx = (as.roundRobinIdx + 1) % len(valid)
	chosen := valid[as.roundRobinIdx]
	// Speculative affinity bind removed here too — see the sticky branch
	// above for why.
	return chosen, nil
}

// GetOrAssignStickyAccount guarantees returning the assigned sticky account for an API key.
// If not yet bound, it deterministically picks the best non-exhausted account and pins it permanently to the key.
func (as *AccountSelector) GetOrAssignStickyAccount(apiKey string, tenant TenantConfig) *AccountRecord {
	accounts := globalAccountManager.GetAccounts()

	as.mu.Lock()
	defer as.mu.Unlock()

	if len(accounts) == 0 {
		return nil
	}

	// 1. Explicitly pinned in tenant config
	if tenant.StickyAccount != "" {
		for _, acc := range accounts {
			if acc.Email == tenant.StickyAccount && !acc.Disabled {
				return acc
			}
		}
	}

	// 2. Previously assigned sticky account for this client API key
	if apiKey != "" {
		if assignedEmail, ok := as.stickyMap[apiKey]; ok {
			for _, acc := range accounts {
				if acc.Email == assignedEmail && !acc.Disabled && hasRemainingQuota(acc) {
					return acc
				}
			}
		}
	}

	// 3. Select best available non-exhausted account
	var bestAcc *AccountRecord
	var maxQuota float64 = -1.0

	for _, acc := range accounts {
		if acc.Disabled {
			continue
		}
		q := getEffectiveQuotaFraction(acc)
		if q > maxQuota {
			maxQuota = q
			bestAcc = acc
		}
	}

	if bestAcc != nil && maxQuota > 0 {
		if apiKey != "" {
			as.stickyMap[apiKey] = bestAcc.Email
		}
		return bestAcc
	}

	// 4. Fallback: first non-disabled account
	for _, acc := range accounts {
		if !acc.Disabled {
			if apiKey != "" {
				as.stickyMap[apiKey] = acc.Email
			}
			return acc
		}
	}

	return accounts[0]
}

// GetAffinityAccount returns the assigned email for a conversation anchor or
// session id. A pure peek — doesn't refresh the LRU position or TTL, since
// callers use this to inspect the binding without necessarily acting on it.
func (as *AccountSelector) GetAffinityAccount(conversationKey string) string {
	if conversationKey == "" {
		return ""
	}
	as.mu.Lock()
	defer as.mu.Unlock()
	if el, ok := as.affinityIdx[conversationKey]; ok {
		entry := el.Value.(*affinityLRUEntry)
		if time.Now().Before(entry.expiresAt) {
			return entry.email
		}
	}
	return ""
}

// RememberAffinity binds a conversation anchor to an account upon successful response.
func (as *AccountSelector) RememberAffinity(conversationKey, email string) {
	if conversationKey == "" || email == "" {
		return
	}
	as.mu.Lock()
	defer as.mu.Unlock()
	as.bindAffinityLocked(conversationKey, email, time.Now())
}

func (as *AccountSelector) bindAffinityLocked(key, email string, now time.Time) {
	if key == "" || email == "" {
		return
	}

	// Already bound: update in place and promote to most-recently-used.
	if el, ok := as.affinityIdx[key]; ok {
		entry := el.Value.(*affinityLRUEntry)
		entry.email = email
		entry.expiresAt = now.Add(sessionAffinityTTL)
		as.affinityLRU.MoveToFront(el)
		return
	}

	// New binding: make room first. Evict from the real tail (truly
	// least-recently-used, not a random map entry) — expired ones first,
	// since those are free wins, then plain LRU order if still full.
	for len(as.affinityIdx) >= maxSessionAffinityEntries {
		back := as.affinityLRU.Back()
		if back == nil {
			break
		}
		be := back.Value.(*affinityLRUEntry)
		as.affinityLRU.Remove(back)
		delete(as.affinityIdx, be.key)
	}

	entry := &affinityLRUEntry{key: key, email: email, expiresAt: now.Add(sessionAffinityTTL)}
	as.affinityIdx[key] = as.affinityLRU.PushFront(entry)
}

func hasRemainingQuota(acc *AccountRecord) bool {
	// If account is under cooldown due to quota exhaustion, check if resetTime has arrived
	if !acc.CooldownUntil.IsZero() && time.Now().Before(acc.CooldownUntil) {
		return false
	}

	if len(acc.Quotas) == 0 {
		return true // assume ok if unprobed
	}
	now := time.Now()
	for _, grp := range acc.Quotas {
		if strings.Contains(strings.ToLower(grp.DisplayName), "gemini") {
			for _, b := range grp.Buckets {
				if b.RemainingFraction <= 0.05 {
					// This bucket's own fraction is a snapshot from the
					// last FetchQuota — nothing re-fetches it on a timer
					// (RefreshAllQuotas only runs off a manual dashboard
					// button), so without checking the bucket's own
					// ResetTime here, an account stayed excluded from the
					// pool forever after its window reset, until someone
					// happened to click refresh. CooldownUntil above is the
					// same ResetTime baked in at the time of that last
					// fetch, so it goes stale in exactly the same way.
					if b.ResetTime != "" {
						if t, err := time.Parse(time.RFC3339Nano, b.ResetTime); err == nil && now.After(t) {
							continue
						}
						if t, err := time.Parse(time.RFC3339, b.ResetTime); err == nil && now.After(t) {
							continue
						}
					}
					return false
				}
			}
		}
	}
	return true
}

func getEffectiveQuotaFraction(acc *AccountRecord) float64 {
	if !acc.CooldownUntil.IsZero() && time.Now().Before(acc.CooldownUntil) {
		return 0.0
	}
	if len(acc.Quotas) == 0 {
		return 0.5
	}
	// Same "claude"-group exclusion as FetchQuota's CooldownUntil calc and
	// pickFiveHourAndWeekly below — a free-tier account's 3P (Claude/GPT)
	// bucket can legitimately sit at 0% for days while Gemini itself is
	// still wide open. Without filtering to the Gemini group here, that
	// unrelated 3P bucket dragged minFrac to ~0 for nearly every real
	// account, making this ranking signal meaningless (every account looks
	// equally "almost exhausted" regardless of its actual Gemini quota).
	minFrac := 1.0
	for _, grp := range acc.Quotas {
		if strings.Contains(strings.ToLower(grp.DisplayName), "claude") {
			continue
		}
		for _, b := range grp.Buckets {
			frac := b.RemainingFraction
			if frac <= 0.05 && b.ResetTime != "" {
				if t, err := time.Parse(time.RFC3339Nano, b.ResetTime); err == nil && time.Now().After(t) {
					frac = 1.0
				} else if t, err := time.Parse(time.RFC3339, b.ResetTime); err == nil && time.Now().After(t) {
					frac = 1.0
				}
			}
			if frac < minFrac {
				minFrac = frac
			}
		}
	}
	return minFrac
}

// UnstickAccount removes any sticky binding pointing to the exhausted email
func (as *AccountSelector) UnstickAccount(email string) {
	as.mu.Lock()
	defer as.mu.Unlock()
	for k, v := range as.stickyMap {
		if v == email {
			delete(as.stickyMap, k)
		}
	}
	for el := as.affinityLRU.Front(); el != nil; {
		next := el.Next()
		entry := el.Value.(*affinityLRUEntry)
		if entry.email == email {
			as.affinityLRU.Remove(el)
			delete(as.affinityIdx, entry.key)
		}
		el = next
	}
}
