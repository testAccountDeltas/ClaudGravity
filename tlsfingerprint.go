// Go's crypto/tls emits a ClientHello that no shipping browser or CLI
// produces, so JA3/JA4-based fingerprinting can single out this gateway's
// outbound traffic to Google before a single byte of the actual request is
// inspected. This file gives outbound calls to Google (CloudCode/Antigravity,
// OAuth, Gemini Developer API) a ClientHello matching a real Chrome browser
// instead, via uTLS (github.com/refraction-networking/utls).
//
// Ported (simplified — no proxy/SOCKS5 support, since this gateway dials
// Google directly, never through an upstream proxy of its own) from the
// independent clirelay project's internal/tlsfingerprint package, which
// solves the exact same problem for the exact same upstream.
package main

import (
	"bufio"
	"context"
	"crypto/x509"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	utls "github.com/refraction-networking/utls"
	"golang.org/x/net/http2"
)

const fingerprintDialTimeout = 30 * time.Second

// Health-check bounds for pooled HTTP/2 connections — matches the values
// clirelay settled on for this exact Google-facing traffic: ReadIdleTimeout
// detects a peer that silently stopped answering (NAT/load-balancer drop
// without FIN/RST), PingTimeout bounds the PONG wait before tearing the
// connection down, and IdleConnTimeout bounds how long an unused connection
// is kept around at all.
const (
	h2ReadIdleTimeout = 30 * time.Second
	h2PingTimeout     = 15 * time.Second
	h2IdleConnTimeout = 90 * time.Second
)

// fingerprintedRoundTripper performs HTTPS requests over connections whose
// ClientHello matches a real browser, managing its own HTTP/2 connections
// since Go only routes a connection to its HTTP/2 stack when it's a
// *crypto/tls.Conn — a utls connection never is, so handing it to
// http.Transport would silently downgrade to HTTP/1.1 even after the server
// agreed to HTTP/2 via ALPN.
type fingerprintedRoundTripper struct {
	helloID utls.ClientHelloID
	rootCAs *x509.CertPool // nil uses the system pool

	mu      sync.Mutex
	h2Conns map[string]*http2.ClientConn
	dialing map[string]*sync.Cond
}

// newFingerprintedRoundTripper builds a RoundTripper that presents a Chrome
// TLS fingerprint on every handshake. Chrome is the least remarkable choice
// on the public internet.
func newFingerprintedRoundTripper() *fingerprintedRoundTripper {
	return &fingerprintedRoundTripper{
		helloID: utls.HelloChrome_Auto,
		h2Conns: make(map[string]*http2.ClientConn),
		dialing: make(map[string]*sync.Cond),
	}
}

func (t *fingerprintedRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.URL == nil {
		return nil, errors.New("tlsfingerprint: request has no URL")
	}
	if !strings.EqualFold(req.URL.Scheme, "https") {
		return nil, fmt.Errorf("tlsfingerprint: scheme %q is not supported, want https", req.URL.Scheme)
	}

	host := req.URL.Hostname()
	addr := canonicalAddr(req.URL)

	h2Conn, err := t.h2ConnFor(req.Context(), host, addr)
	if err != nil {
		return nil, err
	}
	if h2Conn != nil {
		resp, errRT := h2Conn.RoundTrip(req)
		if errRT != nil {
			t.dropH2Conn(addr, h2Conn)
			return nil, errRT
		}
		return resp, nil
	}

	// ALPN didn't select h2 for this host — single-use HTTP/1.1 exchange, no
	// keep-alive pool behind it. Every Google endpoint this gateway talks to
	// negotiates HTTP/2, so this path is a rarely-hit fallback, not the norm.
	return t.roundTripHTTP1(req, host, addr)
}

func (t *fingerprintedRoundTripper) h2ConnFor(ctx context.Context, host, addr string) (*http2.ClientConn, error) {
	t.mu.Lock()
	for {
		if conn, ok := t.h2Conns[addr]; ok {
			if conn.CanTakeNewRequest() {
				t.mu.Unlock()
				return conn, nil
			}
			delete(t.h2Conns, addr)
		}
		cond, inFlight := t.dialing[addr]
		if !inFlight {
			break
		}
		cond.Wait()
	}

	cond := sync.NewCond(&t.mu)
	t.dialing[addr] = cond
	t.mu.Unlock()

	conn, err := t.dialH2(ctx, host, addr)

	t.mu.Lock()
	delete(t.dialing, addr)
	cond.Broadcast()
	if err == nil && conn != nil {
		t.h2Conns[addr] = conn
	}
	t.mu.Unlock()

	return conn, err
}

// dialH2 establishes a fingerprinted TLS connection and, when ALPN selected
// HTTP/2, wraps it as an HTTP/2 client connection. A nil connection with a
// nil error means the peer chose HTTP/1.1.
func (t *fingerprintedRoundTripper) dialH2(ctx context.Context, host, addr string) (*http2.ClientConn, error) {
	tlsConn, err := t.dialTLS(ctx, host, addr)
	if err != nil {
		return nil, err
	}
	if tlsConn.ConnectionState().NegotiatedProtocol != http2.NextProtoTLS {
		_ = tlsConn.Close()
		return nil, nil
	}
	h2Transport := &http2.Transport{
		ReadIdleTimeout: h2ReadIdleTimeout,
		PingTimeout:     h2PingTimeout,
		IdleConnTimeout: h2IdleConnTimeout,
	}
	h2Conn, err := h2Transport.NewClientConn(tlsConn)
	if err != nil {
		_ = tlsConn.Close()
		return nil, err
	}
	return h2Conn, nil
}

func (t *fingerprintedRoundTripper) dialTLS(ctx context.Context, host, addr string) (*utls.UConn, error) {
	dialer := &net.Dialer{Timeout: fingerprintDialTimeout, KeepAlive: 30 * time.Second}
	rawConn, err := dialer.DialContext(ctx, "tcp", addr)
	if err != nil {
		return nil, err
	}

	if deadline, ok := ctx.Deadline(); ok {
		_ = rawConn.SetDeadline(deadline)
	} else {
		_ = rawConn.SetDeadline(time.Now().Add(fingerprintDialTimeout))
	}

	tlsConn := utls.UClient(rawConn, &utls.Config{
		ServerName: host,
		RootCAs:    t.rootCAs,
	}, t.helloID)
	if err = tlsConn.HandshakeContext(ctx); err != nil {
		_ = rawConn.Close()
		return nil, err
	}
	_ = rawConn.SetDeadline(time.Time{}) // clear: streaming responses must not inherit the handshake deadline
	return tlsConn, nil
}

func (t *fingerprintedRoundTripper) roundTripHTTP1(req *http.Request, host, addr string) (*http.Response, error) {
	tlsConn, err := t.dialTLS(req.Context(), host, addr)
	if err != nil {
		return nil, err
	}
	if err = req.Write(tlsConn); err != nil {
		_ = tlsConn.Close()
		return nil, err
	}
	resp, err := http.ReadResponse(bufio.NewReader(tlsConn), req)
	if err != nil {
		_ = tlsConn.Close()
		return nil, err
	}
	resp.Body = &fingerprintConnClosingBody{ReadCloser: resp.Body, conn: tlsConn}
	return resp, nil
}

// fingerprintConnClosingBody ties a response body to the connection that
// produced it, since this pool-less HTTP/1.1 path has nothing else closing it.
type fingerprintConnClosingBody struct {
	io.ReadCloser
	conn net.Conn
}

func (b *fingerprintConnClosingBody) Close() error {
	err := b.ReadCloser.Close()
	if errConn := b.conn.Close(); err == nil {
		err = errConn
	}
	return err
}

func (t *fingerprintedRoundTripper) dropH2Conn(addr string, conn *http2.ClientConn) {
	t.mu.Lock()
	if cached, ok := t.h2Conns[addr]; ok && cached == conn {
		delete(t.h2Conns, addr)
	}
	t.mu.Unlock()
}

// CloseIdleConnections implements the optional http.Transport behaviour used
// by http.Client.CloseIdleConnections.
func (t *fingerprintedRoundTripper) CloseIdleConnections() {
	t.mu.Lock()
	conns := make([]*http2.ClientConn, 0, len(t.h2Conns))
	for host, conn := range t.h2Conns {
		conns = append(conns, conn)
		delete(t.h2Conns, host)
	}
	t.mu.Unlock()
	for _, conn := range conns {
		_ = conn.Close()
	}
}

func canonicalAddr(u *url.URL) string {
	host := u.Hostname()
	port := u.Port()
	if port == "" {
		port = "443"
	}
	return net.JoinHostPort(host, port)
}
