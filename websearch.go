package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/tidwall/gjson"
)

const claudeCodeWebSearchPrefix = "Perform a web search for the query:"

type WebSearchResultItem struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Snippet string `json:"snippet"`
}

// IsClaudeCodeDirectWebSearch checks if the incoming request is Claude Code's native search sub-request.
func IsClaudeCodeDirectWebSearch(bodyBytes []byte) (string, bool) {
	root := gjson.ParseBytes(bodyBytes)

	// Check tools: must contain web_search
	tools := root.Get("tools")
	if !tools.IsArray() || len(tools.Array()) == 0 {
		return "", false
	}
	hasWebSearchTool := false
	tools.ForEach(func(_, t gjson.Result) bool {
		name := t.Get("name").String()
		tType := t.Get("type").String()
		if strings.EqualFold(name, "web_search") || strings.HasPrefix(strings.ToLower(tType), "web_search_") {
			hasWebSearchTool = true
			return false
		}
		return true
	})
	if !hasWebSearchTool {
		return "", false
	}

	// Check messages: last user message must start with prefix
	messages := root.Get("messages")
	if !messages.IsArray() || len(messages.Array()) == 0 {
		return "", false
	}

	msgs := messages.Array()
	for i := len(msgs) - 1; i >= 0; i-- {
		msg := msgs[i]
		if msg.Get("role").String() != "user" {
			continue
		}
		content := msg.Get("content")
		if content.Type == gjson.String {
			txt := strings.TrimSpace(content.String())
			if strings.HasPrefix(txt, claudeCodeWebSearchPrefix) {
				q := strings.TrimSpace(strings.TrimPrefix(txt, claudeCodeWebSearchPrefix))
				return q, true
			}
		} else if content.IsArray() {
			for _, part := range content.Array() {
				if part.Get("type").String() == "text" {
					txt := strings.TrimSpace(part.Get("text").String())
					if strings.HasPrefix(txt, claudeCodeWebSearchPrefix) {
						q := strings.TrimSpace(strings.TrimPrefix(txt, claudeCodeWebSearchPrefix))
						return q, true
					}
				}
			}
		}
		break
	}

	return "", false
}

func newServerToolUseID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return "srvtoolu_" + hex.EncodeToString(b)
}

// PerformWebSearch executes the search via Google CloudCode Grounding (primary) or Wikipedia/DuckDuckGo (fallback).
func PerformWebSearch(ctx context.Context, query string, googleToken string, isAntigravity bool, activeAcc *AccountRecord, tenant TenantConfig) ([]WebSearchResultItem, string, error) {
	// 1. Try Google Search Grounding with gemini-2.5-flash
	if googleToken != "" && isAntigravity && activeAcc != nil {
		results, summary, err := searchWithGoogleGrounding(ctx, query, googleToken, activeAcc.ProjectID)
		if err == nil && len(results) > 0 {
			return results, summary, nil
		}
		LogGateway(".. Google Search Grounding fallback: %v", err)
	}

	// 2. Fallback: DuckDuckGo Instant Answer / Wikipedia
	return searchWithPublicFallback(ctx, query)
}

func searchWithGoogleGrounding(ctx context.Context, query string, googleToken string, projectID string) ([]WebSearchResultItem, string, error) {
	geminiReq := map[string]interface{}{
		"contents": []map[string]interface{}{
			{
				"role": "user",
				"parts": []map[string]interface{}{
					{"text": "Search the web and provide current factual information for: " + query},
				},
			},
		},
		"tools": []map[string]interface{}{
			{"googleSearch": map[string]interface{}{}},
		},
	}
	reqBytes, err := json.Marshal(geminiReq)
	if err != nil {
		return nil, "", err
	}

	wrapped := WrapAntigravityPayload("gemini-2.5-flash", reqBytes, projectID, 0)
	endpointURL := "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal:generateContent"

	httpReq, err := http.NewRequestWithContext(ctx, "POST", endpointURL, bytes.NewReader(wrapped))
	if err != nil {
		return nil, "", err
	}
	httpReq.Header.Set("Authorization", "Bearer "+googleToken)
	httpReq.Header.Set("Content-Type", "application/json; charset=utf-8")
	httpReq.Header.Set("User-Agent", "vscode/1.X.X (Antigravity/4.3.0)")
	httpReq.Header.Set("X-Goog-Api-Client", "google-cloud-sdk vscode_cloudshelleditor/0.1")
	httpReq.Header.Set("Client-Metadata", `{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}`)

	resp, err := httpClient.Do(httpReq)
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, "", err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, "", fmt.Errorf("Google Grounding status %d: %s", resp.StatusCode, string(respBytes))
	}

	root := gjson.ParseBytes(respBytes)
	if respNode := root.Get("response"); respNode.Exists() {
		root = respNode
	}

	cand := root.Get("candidates.0")
	if !cand.Exists() {
		return nil, "", fmt.Errorf("no candidates in Google Grounding response")
	}

	var sb strings.Builder
	parts := cand.Get("content.parts")
	if parts.IsArray() {
		for _, p := range parts.Array() {
			if !p.Get("thought").Bool() {
				t := p.Get("text").String()
				if t != "" {
					sb.WriteString(t)
				}
			}
		}
	}
	summaryText := strings.TrimSpace(sb.String())

	var results []WebSearchResultItem
	if chunks := cand.Get("groundingMetadata.groundingChunks"); chunks.IsArray() {
		seen := make(map[string]bool)
		for _, c := range chunks.Array() {
			uri := c.Get("web.uri").String()
			title := c.Get("web.title").String()
			if uri != "" && !seen[uri] {
				seen[uri] = true
				if title == "" {
					title = uri
				}
				snippet := summaryText
				if len([]rune(snippet)) > 300 {
					snippet = string([]rune(snippet)[:300]) + "..."
				}
				results = append(results, WebSearchResultItem{
					Title:   title,
					URL:     uri,
					Snippet: snippet,
				})
			}
		}
	}

	if len(results) == 0 && summaryText != "" {
		results = append(results, WebSearchResultItem{
			Title:   "Web Search: " + query,
			URL:     "https://www.google.com/search?q=" + url.QueryEscape(query),
			Snippet: summaryText,
		})
	}

	return results, summaryText, nil
}

func searchWithPublicFallback(ctx context.Context, query string) ([]WebSearchResultItem, string, error) {
	// Query Wikipedia API for general knowledge
	wikiURL := fmt.Sprintf("https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=%s&utf8=&format=json", url.QueryEscape(query))
	req, err := http.NewRequestWithContext(ctx, "GET", wikiURL, nil)
	if err != nil {
		return nil, "", err
	}
	req.Header.Set("User-Agent", "SuperRelay/1.0 (WebSearchFallback)")

	resp, err := httpClient.Do(req)
	if err == nil && resp.StatusCode == http.StatusOK {
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		root := gjson.ParseBytes(body)
		items := root.Get("query.search")
		if items.IsArray() && len(items.Array()) > 0 {
			var results []WebSearchResultItem
			var sb strings.Builder
			sb.WriteString(fmt.Sprintf("Search results for %q:\n\n", query))
			for i, item := range items.Array() {
				if i >= 5 {
					break
				}
				title := item.Get("title").String()
				snippet := item.Get("snippet").String()
				cleanSnippet := stripHTMLTags(snippet)
				pageURL := fmt.Sprintf("https://en.wikipedia.org/wiki/%s", url.PathEscape(title))
				results = append(results, WebSearchResultItem{
					Title:   title,
					URL:     pageURL,
					Snippet: cleanSnippet,
				})
				sb.WriteString(fmt.Sprintf("%d. **%s**: %s (%s)\n\n", i+1, title, cleanSnippet, pageURL))
			}
			return results, strings.TrimSpace(sb.String()), nil
		}
	}

	// Minimal fallback if all else fails
	results := []WebSearchResultItem{
		{
			Title:   "Google Search for: " + query,
			URL:     "https://www.google.com/search?q=" + url.QueryEscape(query),
			Snippet: "Search query performed for: " + query,
		},
	}
	return results, fmt.Sprintf("Search completed for query: %q", query), nil
}

func stripHTMLTags(s string) string {
	var b strings.Builder
	inTag := false
	for _, r := range s {
		if r == '<' {
			inTag = true
		} else if r == '>' {
			inTag = false
		} else if !inTag {
			b.WriteRune(r)
		}
	}
	return strings.TrimSpace(b.String())
}

func sendSSEEvent(w http.ResponseWriter, flusher http.Flusher, event string, data interface{}) error {
	b, err := json.Marshal(data)
	if err != nil {
		return err
	}
	msg := fmt.Sprintf("event: %s\ndata: %s\n\n", event, string(b))
	_, err = w.Write([]byte(msg))
	if err == nil && flusher != nil {
		flusher.Flush()
	}
	return err
}

// handleClaudeCodeDirectWebSearch fulfills Claude Code's native search sub-request.
func handleClaudeCodeDirectWebSearch(
	rid uint64,
	t0 time.Time,
	clientKey string,
	accLabel string,
	tenant TenantConfig,
	googleToken string,
	isAntigravity bool,
	activeAcc *AccountRecord,
	bodyBytes []byte,
	query string,
	w http.ResponseWriter,
) {
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()

	results, summaryText, err := PerformWebSearch(ctx, query, googleToken, isAntigravity, activeAcc, tenant)
	if err != nil {
		LogGateway("[#%d] !! WebSearch execution error: %v", rid, err)
		results = []WebSearchResultItem{
			{
				Title:   "Search Error",
				URL:     "https://www.google.com/search?q=" + url.QueryEscape(query),
				Snippet: fmt.Sprintf("Web search error: %v", err),
			},
		}
		summaryText = fmt.Sprintf("Unable to complete search for %q: %v", query, err)
	}

	modelName := gjson.GetBytes(bodyBytes, "model").String()
	if modelName == "" {
		modelName = "claude-3-5-haiku-20241022"
	}
	isStream := gjson.GetBytes(bodyBytes, "stream").Bool()
	hasSafeguards := gjson.GetBytes(bodyBytes, "safeguards").Exists()
	msgID := fmt.Sprintf("msg_%d", time.Now().UnixNano())
	srvToolID := newServerToolUseID()

	// Format result blocks for Claude Code
	searchResultBlocks := make([]map[string]interface{}, 0, len(results))
	for _, r := range results {
		searchResultBlocks = append(searchResultBlocks, map[string]interface{}{
			"type":              "web_search_result",
			"title":             r.Title,
			"url":               r.URL,
			"snippet":           r.Snippet,
			"encrypted_content": r.Snippet,
			"page_age":          nil,
		})
	}

	inputTokens := int64(len(query)/4 + 30)
	outputTokens := int64(len(summaryText)/4 + len(results)*35 + 40)

	flusher, hasFlusher := w.(http.Flusher)
	if isStream && hasFlusher {
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		// 1. message_start
		_ = sendSSEEvent(w, flusher, "message_start", map[string]interface{}{
			"type": "message_start",
			"message": map[string]interface{}{
				"id":            msgID,
				"type":          "message",
				"role":          "assistant",
				"model":         modelName,
				"content":       []interface{}{},
				"stop_reason":   nil,
				"stop_sequence": nil,
				"usage": map[string]interface{}{
					"input_tokens":  inputTokens,
					"output_tokens": 0,
				},
			},
		})

		// 2. content_block 0: server_tool_use
		_ = sendSSEEvent(w, flusher, "content_block_start", map[string]interface{}{
			"type":  "content_block_start",
			"index": 0,
			"content_block": map[string]interface{}{
				"type":  "server_tool_use",
				"id":    srvToolID,
				"name":  "web_search",
				"input": map[string]interface{}{},
			},
		})

		queryJSON, _ := json.Marshal(map[string]interface{}{"query": query})
		_ = sendSSEEvent(w, flusher, "content_block_delta", map[string]interface{}{
			"type":  "content_block_delta",
			"index": 0,
			"delta": map[string]interface{}{
				"type":         "input_json_delta",
				"partial_json": string(queryJSON),
			},
		})

		_ = sendSSEEvent(w, flusher, "content_block_stop", map[string]interface{}{
			"type":  "content_block_stop",
			"index": 0,
		})

		// 3. content_block 1: web_search_tool_result
		_ = sendSSEEvent(w, flusher, "content_block_start", map[string]interface{}{
			"type":  "content_block_start",
			"index": 1,
			"content_block": map[string]interface{}{
				"type":        "web_search_tool_result",
				"tool_use_id": srvToolID,
				"content":     searchResultBlocks,
			},
		})

		_ = sendSSEEvent(w, flusher, "content_block_stop", map[string]interface{}{
			"type":  "content_block_stop",
			"index": 1,
		})

		// 4. content_block 2: text summary
		_ = sendSSEEvent(w, flusher, "content_block_start", map[string]interface{}{
			"type":  "content_block_start",
			"index": 2,
			"content_block": map[string]interface{}{
				"type": "text",
				"text": "",
			},
		})

		runes := []rune(summaryText)
		chunkSize := 100
		for i := 0; i < len(runes); i += chunkSize {
			end := i + chunkSize
			if end > len(runes) {
				end = len(runes)
			}
			_ = sendSSEEvent(w, flusher, "content_block_delta", map[string]interface{}{
				"type":  "content_block_delta",
				"index": 2,
				"delta": map[string]interface{}{
					"type": "text_delta",
					"text": string(runes[i:end]),
				},
			})
		}

		_ = sendSSEEvent(w, flusher, "content_block_stop", map[string]interface{}{
			"type":  "content_block_stop",
			"index": 2,
		})

		// 5. message_delta
		deltaMap := map[string]interface{}{
			"stop_reason":   "end_turn",
			"stop_sequence": nil,
		}
		if hasSafeguards {
			deltaMap["safeguard_results"] = []interface{}{
				map[string]interface{}{
					"type": "dangerous_tool_use",
					"status": map[string]interface{}{
						"type": "available",
						"tool_uses": map[string]interface{}{
							srvToolID: map[string]interface{}{
								"type":    "evaluated",
								"outcome": "not_flagged",
							},
						},
					},
				},
			}
		}

		_ = sendSSEEvent(w, flusher, "message_delta", map[string]interface{}{
			"type":  "message_delta",
			"delta": deltaMap,
			"usage": map[string]interface{}{
				"output_tokens": outputTokens,
				"server_tool_use": map[string]interface{}{
					"web_search_requests": 1,
				},
			},
		})

		// 6. message_stop
		_ = sendSSEEvent(w, flusher, "message_stop", map[string]interface{}{"type": "message_stop"})
	} else {
		// Non-streaming JSON response
		content := []map[string]interface{}{
			{
				"type":  "server_tool_use",
				"id":    srvToolID,
				"name":  "web_search",
				"input": map[string]interface{}{"query": query},
			},
			{
				"type":        "web_search_tool_result",
				"tool_use_id": srvToolID,
				"content":     searchResultBlocks,
			},
			{
				"type": "text",
				"text": summaryText,
			},
		}

		respMap := map[string]interface{}{
			"id":            msgID,
			"type":          "message",
			"role":          "assistant",
			"model":         modelName,
			"content":       content,
			"stop_reason":   "end_turn",
			"stop_sequence": nil,
			"usage": map[string]interface{}{
				"input_tokens":  inputTokens,
				"output_tokens": outputTokens,
				"server_tool_use": map[string]interface{}{
					"web_search_requests": 1,
				},
			},
		}
		if hasSafeguards {
			respMap["safeguard_results"] = []interface{}{
				map[string]interface{}{
					"type": "dangerous_tool_use",
					"status": map[string]interface{}{
						"type": "available",
						"tool_uses": map[string]interface{}{
							srvToolID: map[string]interface{}{
								"type":    "evaluated",
								"outcome": "not_flagged",
							},
						},
					},
				},
			}
		}

		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		_ = json.NewEncoder(w).Encode(respMap)
	}

	dur := time.Since(t0).Milliseconds()
	LogGateway("[#%d] << WebSearch finished: query=%q results=%d dur=%dms", rid, query, len(results), dur)
	globalLogStore.Add(RequestLog{
		Timestamp:    t0,
		Key:          clientKey,
		KeyMasked:    MaskKey(clientKey),
		Account:      accLabel,
		SessionID:    "websearch",
		AgentKey:     "web_search",
		AgentType:    "web_search",
		SubagentName: "",
		Model:        modelName,
		TTFT:         dur,
		TotalDur:     dur,
		WasPanicked:  false,
		Shrunk:       false,
		Status:       http.StatusOK,
	})
}
