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
func DeriveConversationAnchor(bodyBytes []byte, headerSessionID string) string {
	if headerSessionID != "" {
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
	as.mu.Lock()
	defer as.mu.Unlock()

	accounts := globalAccountManager.GetAccounts()
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

	// 3. Fallback: Sticky per API Key
	if (tenant.Strategy == "sticky" || tenant.Strategy == "") && apiKey != "" {
		if assignedEmail, ok := as.stickyMap[apiKey]; ok {
			for _, acc := range accounts {
				if acc.Email == assignedEmail && !isExcluded(acc) && hasRemainingQuota(acc) {
					as.bindAffinityLocked(conversationKey, acc.Email, now)
					return acc, nil
				}
			}
		}
	}

	// 4. Select the best available account based on remaining quota
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

	if bestAcc != nil {
		if apiKey != "" {
			as.stickyMap[apiKey] = bestAcc.Email
		}
		as.bindAffinityLocked(conversationKey, bestAcc.Email, now)
		return bestAcc, nil
	}

	// 5. Fallback: Round-Robin across any non-excluded accounts
	valid := make([]*AccountRecord, 0)
	for _, acc := range accounts {
		if !isExcluded(acc) {
			valid = append(valid, acc)
		}
	}

	if len(valid) == 0 {
		return nil, fmt.Errorf("all accounts are disabled, cooling down or excluded")
	}

	as.roundRobinIdx = (as.roundRobinIdx + 1) % len(valid)
	chosen := valid[as.roundRobinIdx]
	as.bindAffinityLocked(conversationKey, chosen.Email, now)
	return chosen, nil
}

// GetOrAssignStickyAccount guarantees returning the assigned sticky account for an API key.
// If not yet bound, it deterministically picks the best non-exhausted account and pins it permanently to the key.
func (as *AccountSelector) GetOrAssignStickyAccount(apiKey string, tenant TenantConfig) *AccountRecord {
	as.mu.Lock()
	defer as.mu.Unlock()

	accounts := globalAccountManager.GetAccounts()
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

	if bestAcc != nil {
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
	for _, grp := range acc.Quotas {
		if strings.Contains(strings.ToLower(grp.DisplayName), "gemini") {
			for _, b := range grp.Buckets {
				if b.RemainingFraction <= 0.05 {
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
	minFrac := 1.0
	for _, grp := range acc.Quotas {
		for _, b := range grp.Buckets {
			if b.RemainingFraction < minFrac {
				minFrac = b.RemainingFraction
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
