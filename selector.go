package main

import (
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

type SessionAffinityEntry struct {
	Email     string
	ExpiresAt time.Time
}

type AccountSelector struct {
	mu            sync.Mutex
	stickyMap     map[string]string               // apiKey -> assigned email (fallback)
	affinityMap   map[string]SessionAffinityEntry // conversationAnchor / sessionID -> assigned email
	roundRobinIdx int
}

var globalSelector = &AccountSelector{
	stickyMap:   make(map[string]string),
	affinityMap: make(map[string]SessionAffinityEntry),
}

// DeriveConversationAnchor creates a stable identifier for a multi-turn conversation.
// Every turn of the same conversation shares the same anchor (first user message + system prefix),
// allowing all turns to hit the exact same Google account for 100% prompt caching.
func DeriveConversationAnchor(bodyBytes []byte) string {
	root := gjson.ParseBytes(bodyBytes)

	// If client explicitly sent a session id, use it
	if sid := root.Get("metadata.session_id").String(); sid != "" {
		return "sid:" + sid
	}

	// Find the first user message
	firstUserMsg := ""
	msgs := root.Get("messages")
	if msgs.IsArray() {
		for _, m := range msgs.Array() {
			if m.Get("role").String() == "user" {
				c := m.Get("content")
				if c.Type == gjson.String {
					firstUserMsg = c.String()
				} else if c.IsArray() {
					for _, part := range c.Array() {
						if part.Get("type").String() == "text" {
							firstUserMsg = part.Get("text").String()
							break
						}
					}
				}
				if firstUserMsg != "" {
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
		if entry, ok := as.affinityMap[conversationKey]; ok {
			if now.Before(entry.ExpiresAt) {
				for _, acc := range accounts {
					if acc.Email == entry.Email && !isExcluded(acc) && hasRemainingQuota(acc) {
						// Refresh lease
						as.affinityMap[conversationKey] = SessionAffinityEntry{
							Email:     acc.Email,
							ExpiresAt: now.Add(sessionAffinityTTL),
						}
						return acc, nil
					}
				}
			} else {
				delete(as.affinityMap, conversationKey)
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

// GetAffinityAccount returns the assigned email for a conversation anchor or session id.
func (as *AccountSelector) GetAffinityAccount(conversationKey string) string {
	if conversationKey == "" {
		return ""
	}
	as.mu.Lock()
	defer as.mu.Unlock()
	if entry, ok := as.affinityMap[conversationKey]; ok {
		if time.Now().Before(entry.ExpiresAt) {
			return entry.Email
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
	if len(as.affinityMap) >= maxSessionAffinityEntries {
		// Evict expired entries
		for k, e := range as.affinityMap {
			if now.After(e.ExpiresAt) {
				delete(as.affinityMap, k)
			}
		}
		// If still full, drop an arbitrary ~10% (Go map iteration order is
		// randomized, not insertion order, so this isn't actually "oldest" —
		// it's a cheap approximation that's fine for a soft cache eviction)
		if len(as.affinityMap) >= maxSessionAffinityEntries {
			count := 0
			for k := range as.affinityMap {
				delete(as.affinityMap, k)
				count++
				if count >= maxSessionAffinityEntries/10 {
					break
				}
			}
		}
	}
	as.affinityMap[key] = SessionAffinityEntry{
		Email:     email,
		ExpiresAt: now.Add(sessionAffinityTTL),
	}
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
		if grp.DisplayName == "Gemini Models" {
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
	for k, v := range as.affinityMap {
		if v.Email == email {
			delete(as.affinityMap, k)
		}
	}
}
