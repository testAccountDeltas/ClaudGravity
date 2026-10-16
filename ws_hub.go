package main

import (
	"encoding/json"
	"sync"
	"time"

	"golang.org/x/net/websocket"
)

// WSEvent represents a standard JSON message pushed to WebSocket clients.
type WSEvent struct {
	Type      string      `json:"type"`
	Timestamp int64       `json:"timestamp"`
	Data      interface{} `json:"data,omitempty"`
}

// WSClient wraps a single WebSocket connection.
type WSClient struct {
	ws     *websocket.Conn
	send   chan []byte
	closed bool
	mu     sync.Mutex
}

// WSHub coordinates live connections and broadcasts events to all connected clients.
type WSHub struct {
	clients    map[*WSClient]bool
	register   chan *WSClient
	unregister chan *WSClient
	broadcast  chan []byte
	mu         sync.RWMutex
}

var globalWSHub = NewWSHub()

func NewWSHub() *WSHub {
	h := &WSHub{
		clients:    make(map[*WSClient]bool),
		register:   make(chan *WSClient, 32),
		unregister: make(chan *WSClient, 32),
		broadcast:  make(chan []byte, 256),
	}
	go h.run()
	return h
}

func (h *WSHub) run() {
	heartbeat := time.NewTicker(25 * time.Second)
	defer heartbeat.Stop()

	for {
		select {
		case client := <-h.register:
			h.mu.Lock()
			h.clients[client] = true
			h.mu.Unlock()
			// Send initial state snapshot immediately to the new client
			go h.sendInitialSnapshot(client)

		case client := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[client]; ok {
				delete(h.clients, client)
				client.Close()
			}
			h.mu.Unlock()

		case msg := <-h.broadcast:
			h.mu.RLock()
			for client := range h.clients {
				if !client.SafeSend(msg) {
					// Closed or slow client buffer full: unregister to avoid stalling
					go func(c *WSClient) {
						h.unregister <- c
					}(client)
				}
			}
			h.mu.RUnlock()

		case <-heartbeat.C:
			// Ping active clients to keep connection alive through proxies
			pingEvent, _ := json.Marshal(WSEvent{
				Type:      "ping",
				Timestamp: time.Now().UnixMilli(),
			})
			h.mu.RLock()
			for client := range h.clients {
				client.SafeSend(pingEvent)
			}
			h.mu.RUnlock()
		}
	}
}

func (c *WSClient) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.closed {
		c.closed = true
		close(c.send)
		_ = c.ws.Close()
	}
}

// SafeSend queues payload on the client's send channel, holding the same
// mutex Close() uses so a send can never race a close(c.send) and panic.
// Returns false if the client is already closed or its buffer is full
// (slow client) — callers treat both as "drop/unregister", never as a
// reason to send to the channel directly.
func (c *WSClient) SafeSend(payload []byte) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return false
	}
	select {
	case c.send <- payload:
		return true
	default:
		return false
	}
}

func (h *WSHub) Broadcast(eventType string, data interface{}) {
	payload, err := json.Marshal(WSEvent{
		Type:      eventType,
		Timestamp: time.Now().UnixMilli(),
		Data:      data,
	})
	if err != nil {
		return
	}

	select {
	case h.broadcast <- payload:
	default:
		// Broadcast channel full: drop to prevent gateway stalls
	}
}

func (h *WSHub) BroadcastLog(reqLog *RequestLog) {
	if reqLog == nil {
		return
	}
	h.Broadcast("log:entry", reqLog)
}

func (h *WSHub) BroadcastAccounts(accounts []*AccountRecord) {
	h.Broadcast("accounts:update", accounts)
}

func (h *WSHub) BroadcastAccountDeleted(email string) {
	h.Broadcast("account:deleted", map[string]string{"email": email})
}

func (h *WSHub) BroadcastSessionsUpdate() {
	sessions := globalLogStore.QuerySessions(50)
	h.Broadcast("sessions:update", sessions)
}

func (h *WSHub) BroadcastStats(stats interface{}) {
	h.Broadcast("stats:update", stats)
}

func (h *WSHub) sendInitialSnapshot(client *WSClient) {
	accounts := globalAccountManager.GetAccounts()
	sessions := globalLogStore.QuerySessions(50)
	logs, stats := globalLogStore.Query("", "", "", "", false, 100)
	catalog := globalModelCatalog.GetAll()

	snapshot := map[string]interface{}{
		"accounts": accounts,
		"sessions": sessions,
		"logs":     logs,
		"stats":    stats,
		"catalog":  catalog,
	}

	payload, err := json.Marshal(WSEvent{
		Type:      "init",
		Timestamp: time.Now().UnixMilli(),
		Data:      snapshot,
	})
	if err == nil {
		client.SafeSend(payload)
	}
}

// HandleWS upgrades HTTP connection to WebSocket and registers client to hub.
func HandleWS(ws *websocket.Conn) {
	client := &WSClient{
		ws:   ws,
		send: make(chan []byte, 128),
	}
	globalWSHub.register <- client

	// Writer pump: pipes send channel to WebSocket
	go func() {
		defer func() {
			globalWSHub.unregister <- client
		}()
		for msg := range client.send {
			_ = ws.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if _, err := ws.Write(msg); err != nil {
				return
			}
		}
	}()

	// Reader pump: reads control messages / client pings
	defer func() {
		globalWSHub.unregister <- client
	}()

	buf := make([]byte, 1024)
	for {
		_ = ws.SetReadDeadline(time.Now().Add(60 * time.Second))
		n, err := ws.Read(buf)
		if err != nil {
			break
		}
		if n > 0 {
			var incoming struct {
				Type string `json:"type"`
			}
			if err := json.Unmarshal(buf[:n], &incoming); err == nil {
				if incoming.Type == "pong" {
					// Client alive acknowledgment
				}
			}
		}
	}
}
