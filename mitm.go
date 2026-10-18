package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/subtle"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
	"math/big"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// bufferedConn prepends bytes net/http's bufio.Reader already pulled off the
// socket (see HandleConnectTunnel's Hijack() call) ahead of further reads
// from the raw connection, so nothing read during CONNECT header parsing is
// lost to the TLS handshake that follows.
type bufferedConn struct {
	net.Conn
	r io.Reader
}

func (b *bufferedConn) Read(p []byte) (int, error) {
	return b.r.Read(p)
}

// globalMitmCA is set once in main() right after InitMitmCA succeeds, so
// other handlers (the /api/bootstrap installer endpoint in ui.go) can hand
// out the CA's PEM without needing mitmCA threaded through as a parameter.
var globalMitmCA *MitmCA

type MitmCA struct {
	caCert    *x509.Certificate
	caKey     *rsa.PrivateKey
	caPEM     []byte
	caPEMPath string
	certCache sync.Map
	certMu    sync.Mutex
}

func InitMitmCA(caDir string) (*MitmCA, error) {
	if err := os.MkdirAll(caDir, 0755); err != nil {
		return nil, err
	}
	caCertPath := filepath.Join(caDir, "superrelay-ca.pem")
	caKeyPath := filepath.Join(caDir, "superrelay-ca.key")

	var caCert *x509.Certificate
	var caKey *rsa.PrivateKey
	var caPEM []byte

	certBytes, errCert := os.ReadFile(caCertPath)
	keyBytes, errKey := os.ReadFile(caKeyPath)

	if errCert == nil && errKey == nil {
		block, _ := pem.Decode(certBytes)
		if block != nil {
			caCert, _ = x509.ParseCertificate(block.Bytes)
		}
		keyBlock, _ := pem.Decode(keyBytes)
		if keyBlock != nil {
			caKey, _ = x509.ParsePKCS1PrivateKey(keyBlock.Bytes)
		}
		caPEM = certBytes
	}

	if caCert == nil || caKey == nil {
		priv, err := rsa.GenerateKey(rand.Reader, 2048)
		if err != nil {
			return nil, err
		}
		template := x509.Certificate{
			SerialNumber: big.NewInt(time.Now().UnixNano()),
			Subject: pkix.Name{
				Organization: []string{"SuperRelay Internal CA"},
				CommonName:   "SuperRelay Local Root CA",
			},
			NotBefore:             time.Now().Add(-24 * time.Hour),
			NotAfter:              time.Now().Add(10 * 365 * 24 * time.Hour), // 10 years
			KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
			BasicConstraintsValid: true,
			IsCA:                  true,
			MaxPathLen:            2,
		}
		der, err := x509.CreateCertificate(rand.Reader, &template, &template, &priv.PublicKey, priv)
		if err != nil {
			return nil, err
		}
		caCert, _ = x509.ParseCertificate(der)
		caKey = priv
		caPEM = pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
		keyPEM := pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(priv)})

		_ = os.WriteFile(caCertPath, caPEM, 0644)
		_ = os.WriteFile(caKeyPath, keyPEM, 0600)
	}

	return &MitmCA{
		caCert:    caCert,
		caKey:     caKey,
		caPEM:     caPEM,
		caPEMPath: caCertPath,
	}, nil
}

// GetCAPEM returns the CA certificate in PEM form, for handing to clients
// that need to trust it (e.g. the /api/bootstrap installer endpoint) rather
// than reading the file off whatever disk the gateway happens to run on.
func (m *MitmCA) GetCAPEM() string {
	return string(m.caPEM)
}

func (m *MitmCA) GetCAPath() string {
	abs, err := filepath.Abs(m.caPEMPath)
	if err == nil {
		return abs
	}
	return m.caPEMPath
}

func (m *MitmCA) GetCertificate(domain string) (tls.Certificate, error) {
	domain = strings.ToLower(domain)
	if val, ok := m.certCache.Load(domain); ok {
		return val.(tls.Certificate), nil
	}

	m.certMu.Lock()
	defer m.certMu.Unlock()
	if val, ok := m.certCache.Load(domain); ok {
		return val.(tls.Certificate), nil
	}

	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return tls.Certificate{}, err
	}
	// domain may be a hostname (goes in DNSNames) or a literal IP (goes in
	// IPAddresses) — TLS hostname verification checks the IP SAN list, not
	// DNSNames, when the client dials a bare IP (e.g. a remote server's
	// public address), so a string match alone would still fail there.
	dnsNames := []string{"localhost"}
	ips := []net.IP{net.ParseIP("127.0.0.1"), net.ParseIP("::1")}
	if ip := net.ParseIP(domain); ip != nil {
		ips = append(ips, ip)
	} else {
		dnsNames = append(dnsNames, domain)
	}

	serialNumberLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	serialNumber, err := rand.Int(rand.Reader, serialNumberLimit)
	if err != nil {
		serialNumber = big.NewInt(time.Now().UnixNano())
	}

	template := x509.Certificate{
		SerialNumber: serialNumber,
		Subject: pkix.Name{
			CommonName:   domain,
			Organization: []string{"SuperRelay Internal Proxy"},
		},
		NotBefore:             time.Now().Add(-24 * time.Hour),
		NotAfter:              time.Now().Add(365 * 24 * time.Hour),
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		DNSNames:              dnsNames,
		IPAddresses:           ips,
	}
	der, err := x509.CreateCertificate(rand.Reader, &template, m.caCert, &priv.PublicKey, m.caKey)
	if err != nil {
		return tls.Certificate{}, err
	}
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(priv)})
	tlsCert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		return tls.Certificate{}, err
	}
	m.certCache.Store(domain, tlsCert)
	return tlsCert, nil
}

// proxyAuthOK accepts the secret either as "Bearer <secret>" (manual
// testing, curl --proxy-header) or as HTTP Basic auth with the secret as
// the username (or password) (what HTTP clients send when HTTPS_PROXY is set to
// http://<secret>:x@host:port or http://:<secret>@host:port — the standard way to embed proxy
// credentials in the proxy URL).
func proxyAuthOK(r *http.Request, secret string) bool {
	header := r.Header.Get("Proxy-Authorization")
	lower := strings.ToLower(header)
	if strings.HasPrefix(lower, "bearer ") {
		bearer := strings.TrimSpace(header[7:])
		return subtle.ConstantTimeCompare([]byte(bearer), []byte(secret)) == 1
	}
	if strings.HasPrefix(lower, "basic ") {
		encoded := strings.TrimSpace(header[6:])
		decoded, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			return false
		}
		user := decoded
		pass := []byte{}
		if idx := bytes.IndexByte(decoded, ':'); idx >= 0 {
			user = decoded[:idx]
			pass = decoded[idx+1:]
		}
		if subtle.ConstantTimeCompare(user, []byte(secret)) == 1 {
			return true
		}
		if len(pass) > 0 && subtle.ConstantTimeCompare(pass, []byte(secret)) == 1 {
			return true
		}
	}
	return false
}

// isAnthropicHost reports whether host is exactly one of the domains this
// gateway intercepts, or a subdomain of one — not merely a string that
// contains it anywhere.
func isAnthropicHost(host string) bool {
	host = strings.ToLower(host)
	// A trailing "." (valid DNS syntax for "fully qualified") must not let
	// a client dodge this match and fall through to the plain tunnel path
	// below, where none of this gateway's response doctoring applies.
	host = strings.TrimSuffix(host, ".")
	for _, domain := range []string{"anthropic.com", "claude.ai"} {
		if host == domain || strings.HasSuffix(host, "."+domain) {
			return true
		}
	}
	return false
}

var cgnatSubnet = &net.IPNet{
	IP:   net.IPv4(100, 64, 0, 0),
	Mask: net.CIDRMask(10, 32),
}

// checkConnectTarget reports whether host resolves to a public, safe address.
// It returns every verified safe IP (IPv4 addresses first — a dual-stack
// DNS answer commonly lists AAAA before A, and a host with no outbound IPv6
// route failed every CONNECT with "network is unreachable" when only the
// first-returned address was ever tried), or an error if the target is
// private, loopback, link-local, unresolvable, or cloud metadata.
func checkConnectTarget(host string) ([]net.IP, error) {
	ips := []net.IP{net.ParseIP(host)}
	if ips[0] == nil {
		resolved, err := net.LookupIP(host)
		if err != nil {
			return nil, err
		}
		ips = resolved
	}
	if len(ips) == 0 {
		return nil, fmt.Errorf("no addresses found for %s", host)
	}
	for _, ip := range ips {
		if ip == nil || ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() || ip.IsMulticast() || ip.IsUnspecified() || (ip.To4() != nil && cgnatSubnet.Contains(ip.To4())) {
			return nil, fmt.Errorf("blocked target address: %s", ip)
		}
	}
	sorted := make([]net.IP, len(ips))
	copy(sorted, ips)
	sort.SliceStable(sorted, func(i, j int) bool {
		return sorted[i].To4() != nil && sorted[j].To4() == nil
	})
	return sorted, nil
}

// HandleConnectTunnel implements native HTTP CONNECT proxying and TLS interception.
// proxySecret, when non-empty, is required as a "Proxy-Authorization: Bearer
// <secret>" header on every CONNECT request — without it, anyone who can
// reach this port could use it as an open relay to the internet.
func HandleConnectTunnel(w http.ResponseWriter, r *http.Request, ca *MitmCA, targetHandler http.Handler, proxySecret string) {
	if proxySecret != "" && !proxyAuthOK(r, proxySecret) {
		w.Header().Set("Proxy-Authenticate", `Basic realm="SuperRelay Proxy"`)
		http.Error(w, "Proxy Authentication Required", http.StatusProxyAuthRequired)
		return
	}

	host := r.Host
	port := "443"
	if h, p, err := net.SplitHostPort(r.Host); err == nil {
		host = h
		port = p
	}

	// This gateway's whole job is proxying to public hosts
	// (api.anthropic.com, github.com, npm/pypi registries, ...) — it has
	// no legitimate reason to ever CONNECT to a private, link-local, or
	// loopback address, so a client that knows (or defeats) the proxy
	// secret can't use this as an SSRF pivot into the host's own internal
	// network or cloud metadata endpoint (169.254.169.254). Checked before
	// Hijack() so a normal HTTP error response still works here.
	var verifiedIPs []net.IP
	if !isAnthropicHost(host) {
		ips, err := checkConnectTarget(host)
		if err != nil {
			LogGateway("[SuperRelay Gateway] Refused CONNECT to target %s: %v", host, err)
			http.Error(w, "Forbidden", http.StatusForbidden)
			return
		}
		verifiedIPs = ips
	}

	hijacker, ok := w.(http.Hijacker)
	if !ok {
		http.Error(w, "Hijacking not supported", http.StatusInternalServerError)
		return
	}
	rawConn, rw, err := hijacker.Hijack()
	if err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	// A client that pipelines its TLS ClientHello right behind the CONNECT
	// request (same TCP read) already has those bytes sitting in net/http's
	// internal bufio.Reader by the time Hijack() returns — discarding that
	// reader here would drop them forever, and the TLS handshake below would
	// then read from an empty, silent socket until it times out.
	var clientConn net.Conn = rawConn
	if rw != nil && rw.Reader != nil && rw.Reader.Buffered() > 0 {
		clientConn = &bufferedConn{Conn: rawConn, r: io.MultiReader(rw.Reader, rawConn)}
	}

	// 1. Intercept Anthropic and Claude API endpoints locally. isAnthropicHost
	// does an exact-or-subdomain match, not a bare substring one — a raw
	// strings.Contains would also match an unrelated host that merely
	// contains "anthropic.com"/"claude.ai" as a substring (e.g.
	// "anthropic.com.attacker.net" or some third party's
	// "notclaude.ai.example.org"), MITM'ing it with this gateway's own
	// self-signed cert instead of leaving it to PassthroughToReal untouched.
	if isAnthropicHost(host) {
		tlsCert, err := ca.GetCertificate(host)
		if err != nil {
			_, _ = clientConn.Write([]byte("HTTP/1.1 500 Internal Server Error\r\n\r\n"))
			clientConn.Close()
			return
		}
		_, _ = clientConn.Write([]byte("HTTP/1.1 200 Connection Established\r\n\r\n"))
		tlsConfig := &tls.Config{
			Certificates: []tls.Certificate{tlsCert},
			NextProtos:   []string{"http/1.1"},
		}
		tlsConn := tls.Server(clientConn, tlsConfig)
		go serveInterceptedTLS(tlsConn, targetHandler)
		return
	}

	// 2. Transparent TCP tunnel for all other external hosts. Dial a
	// pre-verified IP (never re-resolving the hostname) to prevent TOCTOU
	// DNS rebinding SSRF attacks, falling through the rest of the verified
	// list if the first (IPv4-preferred) address is unreachable from this
	// host — e.g. no outbound IPv6 route even though the name also
	// resolves to an AAAA address.
	var targetConn net.Conn
	if len(verifiedIPs) == 0 {
		targetConn, err = net.DialTimeout("tcp", r.Host, 10*time.Second)
	} else {
		for _, ip := range verifiedIPs {
			targetConn, err = net.DialTimeout("tcp", net.JoinHostPort(ip.String(), port), 10*time.Second)
			if err == nil {
				break
			}
		}
	}
	if err != nil {
		LogGateway("[SuperRelay Gateway] Failed to dial tunnel target %s:%s: %v", host, port, err)
		_, _ = clientConn.Write([]byte("HTTP/1.1 502 Bad Gateway\r\n\r\n"))
		clientConn.Close()
		return
	}

	// Connection successfully established to remote target
	_, _ = clientConn.Write([]byte("HTTP/1.1 200 Connection Established\r\n\r\n"))
	// Guard against either side going silent forever and leaking the
	// goroutines/fds indefinitely (e.g. a client that opens a tunnel and
	// never sends anything) — but net.Conn.SetDeadline is an ABSOLUTE
	// deadline, not an idle timeout: it is never reset by Read/Write, so a
	// single SetDeadline call before io.Copy killed any tunnel still
	// actively transferring data (git clone of a large repo, npm install,
	// downloading model weights) the instant 10 minutes of wall-clock time
	// passed, no matter how busy the connection was. idleCopy resets the
	// deadline before every read instead.
	const idleDeadline = 10 * time.Minute
	go func() {
		defer clientConn.Close()
		defer targetConn.Close()
		idleCopy(targetConn, clientConn, idleDeadline)
	}()
	go func() {
		defer clientConn.Close()
		defer targetConn.Close()
		idleCopy(clientConn, targetConn, idleDeadline)
	}()
}

// idleCopy is io.Copy with a true idle timeout: the deadline is pushed
// forward before every read, so it only fires after idleDeadline of
// genuine silence, never on total connection lifetime.
func idleCopy(dst net.Conn, src net.Conn, idleDeadline time.Duration) {
	buf := make([]byte, 32*1024)
	for {
		_ = src.SetReadDeadline(time.Now().Add(idleDeadline))
		n, err := src.Read(buf)
		if n > 0 {
			_ = dst.SetWriteDeadline(time.Now().Add(idleDeadline))
			if _, werr := dst.Write(buf[:n]); werr != nil {
				return
			}
		}
		if err != nil {
			return
		}
	}
}

// mitmIdleDeadline bounds how long serveInterceptedTLS's connection can go
// completely silent before it's torn down — refreshed on every read/write
// instead of set once, so it only fires on genuine inactivity. A single
// absolute SetDeadline here used to kill the connection 30 real wall-clock
// seconds after the handshake no matter what, which is well inside a real
// generation's TTFT (this gateway's own panic/TTFT watchdogs allow up to
// 180s) — any client routed through this intercepted path (HTTPS_PROXY set,
// no ANTHROPIC_BASE_URL) doing real work past 30s got its response torn off
// mid-stream.
const mitmIdleDeadline = 10 * time.Minute

func serveInterceptedTLS(tlsConn *tls.Conn, handler http.Handler) {
	defer tlsConn.Close()
	_ = tlsConn.SetDeadline(time.Now().Add(mitmIdleDeadline))

	defer func() {
		if rec := recover(); rec != nil {
			LogGateway("[SuperRelay Gateway] MITM handler panic (recovered): %v", rec)
		}
	}()

	if err := tlsConn.Handshake(); err != nil {
		LogGateway("[SuperRelay Gateway] MITM TLS handshake failed: %v", err)
		return
	}
	_ = tlsConn.SetDeadline(time.Now().Add(mitmIdleDeadline))

	reader := bufio.NewReader(tlsConn)
	req, err := http.ReadRequest(reader)
	if err != nil {
		LogGateway("[SuperRelay Gateway] MITM request parse failed: %v", err)
		return
	}

	// http.ReadRequest builds req with context.Background() — real, since
	// this path parses raw requests off a hijacked CONNECT tunnel instead of
	// going through http.Server (which would wire req.Context() to cancel
	// on client disconnect automatically). Without this, a client hitting
	// Ctrl+C mid-generation while routed through HTTPS_PROXY (rather than
	// the direct :8900 TLS API) never propagated cancellation to the
	// upstream Google call at all — r.Context() downstream was real but
	// never actually tied to this connection closing.
	// Drain the whole body into memory BEFORE starting the disconnect
	// watcher below. req.Body is a lazy reader pulling from the same
	// `reader` (bufio.Reader) wrapping tlsConn — whatever body bytes
	// weren't already sitting in its buffer from the header read still
	// have to come from a real tlsConn.Read() the first time something
	// consumes req.Body. The watcher goroutine's own tlsConn.Read() raced
	// directly against that on the raw connection, bypassing `reader`
	// entirely, so it could steal a body byte meant for the request parser
	// — silently corrupting any POST body that didn't fit in bufio's
	// initial read (JSON parse failures in handleMessages, confirmed as
	// the actual failure mode for bodies over a few KB).
	bodyBytes, err := io.ReadAll(req.Body)
	if err != nil {
		LogGateway("[SuperRelay Gateway] MITM read body failed: %v", err)
		return
	}
	_ = req.Body.Close()
	req.Body = io.NopCloser(bytes.NewReader(bodyBytes))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		// Only safe to read tlsConn directly now that the body is fully
		// drained above — nothing else will touch `reader`/tlsConn for
		// input again on this one-shot connection (see the comment below).
		// Any read activity here — new bytes or an error/EOF — means the
		// client is done with this tunnel either way, so stop the upstream
		// call.
		buf := make([]byte, 1)
		n, err := tlsConn.Read(buf)
		if err != nil || n == 0 {
			cancel()
		}
	}()
	req = req.WithContext(ctx)

	rw := newMitmResponseWriter(tlsConn)
	handler.ServeHTTP(rw, req)
	rw.Close()
	// Always close after one request-response pair instead of looping for
	// keep-alive reuse: these intercepted calls (oauth/usage, bootstrap) are
	// low-frequency control calls, and a hand-rolled HTTP/1.1 writer sharing
	// one TLS connection across multiple pipelined requests is a needless
	// source of framing bugs that silently kill the tunnel with no log line.
}

// PassthroughToReal forwards a decrypted, intercepted request to the real
// host it was originally addressed to (api.anthropic.com, claude.ai, ...).
// Only /api/oauth/usage and /api/claude_cli/bootstrap are answered locally
// with fabricated data (they depend on a real subscription we don't have);
// every other path under these intercepted domains must reach the real
// servers untouched, or background features (marketplaces, mcp-registry,
// telemetry, first-run connectivity checks, ...) break with a local 404.
func PassthroughToReal(w http.ResponseWriter, r *http.Request) {
	target := *r.URL
	target.Scheme = "https"
	if isAnthropicHost(r.Host) {
		target.Host = r.Host
	} else {
		target.Host = "api.anthropic.com"
	}

	// r.Body is a one-shot io.ReadCloser; http.NewRequest only wires up
	// GetBody automatically for *bytes.Reader/*bytes.Buffer/*strings.Reader,
	// so passing it directly leaves GetBody nil and Go's transport can't
	// rewind-and-retry on a transient connection drop — confirmed live in
	// superrelay.log: "cannot rewind body after connection loss". Buffering
	// it (these are passthrough API calls, not large file transfers) lets
	// the transport retry transparently instead of surfacing a 502.
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "proxy error", http.StatusBadGateway)
		return
	}
	outReq, err := http.NewRequest(r.Method, target.String(), bytes.NewReader(bodyBytes))
	if err != nil {
		http.Error(w, "proxy error", http.StatusBadGateway)
		return
	}
	outReq.Header = r.Header.Clone()
	outReq.Host = r.Host

	resp, err := passthroughClient.Do(outReq)
	if err != nil {
		LogGateway("[SuperRelay Gateway] MITM passthrough to %s%s failed: %v", r.Host, r.URL.Path, err)
		http.Error(w, "upstream unreachable", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	// MitmResponseWriter.Flush() always buffers the whole body and computes
	// its own real Content-Length — but a chunked upstream response has no
	// Content-Length of its own, so blindly copying its hop-by-hop headers
	// through left the stale "Transfer-Encoding: chunked" sitting next to
	// Flush()'s freshly computed Content-Length: two conflicting framing
	// headers on the same response, which RFC 7230 §3.3.2 forbids and which
	// Node.js's strict HTTP/1.1 parser (what Claude Code's CLI runs on)
	// rejects outright instead of just picking one.
	hopByHop := map[string]bool{
		"Connection": true, "Keep-Alive": true, "Transfer-Encoding": true,
		"Trailer": true, "Upgrade": true, "Proxy-Authenticate": true,
		"Proxy-Authorization": true, "Te": true,
	}
	for k, vv := range resp.Header {
		if hopByHop[k] {
			continue
		}
		for _, v := range vv {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

var passthroughClient = &http.Client{
	Timeout: 30 * time.Second,
}

// HandleBootstrapSpoof handles GET /api/claude_cli/bootstrap
func HandleBootstrapSpoof(w http.ResponseWriter, r *http.Request) {
	LogGateway("[SuperRelay Gateway] >>> Intercepted GET /api/claude_cli/bootstrap <<<")
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("anthropic-organization-id", fakeOrgID)
	w.Header().Set("anthropic-workspace-id", fakeWorkspaceID)
	w.WriteHeader(http.StatusOK)

	resp := map[string]interface{}{
		"org_model_default": map[string]interface{}{
			"name":                    "claude-opus-5-5",
			"updated_at":              "2026-01-01T00:00:00Z",
			"data_source":             "org",
			"override_user_selection": false,
		},
		"oauth_account": map[string]interface{}{
			"account_uuid":                 "0bad747f-0000-4000-a000-000000000000",
			"account_email":                "user@claudgravity.local",
			"organization_uuid":            fakeOrgID,
			"organization_name":            "ClaudGravity Workspace",
			"organization_type":            "enterprise",
			"organization_rate_limit_tier": "tier_4",
			"user_rate_limit_tier":         "tier_4",
			"seat_tier":                    "enterprise",
		},
		"additional_model_options": []interface{}{},
		"model_access": []map[string]interface{}{
			{"api_name": "claude-opus-5-5", "entitled": true},
			{"api_name": "claude-opus-5", "entitled": true},
			{"api_name": "claude-opus-4-8", "entitled": true},
			{"api_name": "claude-opus-4-7", "entitled": true},
			{"api_name": "claude-opus-4-6", "entitled": true},
			{"api_name": "claude-opus-4-5", "entitled": true},
			{"api_name": "claude-opus-4-6-thinking", "entitled": true},
			{"api_name": "claude-sonnet-5-5", "entitled": true},
			{"api_name": "claude-sonnet-5", "entitled": true},
			{"api_name": "claude-sonnet-4-6", "entitled": true},
			{"api_name": "claude-sonnet-4-5", "entitled": true},
			{"api_name": "claude-haiku-5-5", "entitled": true},
			{"api_name": "claude-haiku-4-5", "entitled": true},
			{"api_name": "claude-fable-5-1", "entitled": true},
			{"api_name": "claude-mythos-5", "entitled": true},
			{"api_name": "claude-mythos-5-1", "entitled": true},
			{"api_name": "gemini-3.8-flash-tiered", "entitled": true},
			// Older canonical Claude model ids/aliases: a client pinned to one
			// of these via `/model <id>` or `--model` checks this list before
			// switching and fails with "Model not entitled for this
			// organization" if its target isn't here, even though this
			// gateway happily routes any model name through to Gemini.
			{"api_name": "claude-3-7-sonnet-20250219", "entitled": true},
			{"api_name": "claude-3-7-sonnet-latest", "entitled": true},
			{"api_name": "claude-3-5-sonnet-20241022", "entitled": true},
			{"api_name": "claude-3-5-sonnet-latest", "entitled": true},
			{"api_name": "claude-3-5-haiku-20241022", "entitled": true},
			{"api_name": "claude-3-5-haiku-latest", "entitled": true},
			{"api_name": "claude-3-opus-20240229", "entitled": true},
			{"api_name": "gpt-oss-120b-medium", "entitled": true},
		},
	}
	_ = json.NewEncoder(w).Encode(resp)
}

type MitmResponseWriter struct {
	mu          sync.Mutex
	conn        net.Conn
	header      http.Header
	buf         bytes.Buffer
	statusCode  int
	wroteHeader bool
	headersSent bool
	chunked     bool
	sentOffset  int
}

func newMitmResponseWriter(conn net.Conn) *MitmResponseWriter {
	return &MitmResponseWriter{
		conn:       conn,
		header:     make(http.Header),
		statusCode: http.StatusOK,
	}
}

func (w *MitmResponseWriter) Header() http.Header {
	return w.header
}

func (w *MitmResponseWriter) Write(b []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.buf.Write(b)
}

func (w *MitmResponseWriter) WriteHeader(statusCode int) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if !w.wroteHeader {
		w.statusCode = statusCode
		w.wroteHeader = true
	}
}

// sendHeaders writes the status line and headers exactly once. chunked is
// true for a real mid-stream Flush (SSE), where the final body size isn't
// known yet; false for Close's single-shot path, where the handler has
// already returned and the whole body is sitting in w.buf, so a precise
// Content-Length can still be set just like before this type supported
// incremental streaming at all.
func (w *MitmResponseWriter) sendHeaders(chunked bool) {
	w.headersSent = true
	w.chunked = chunked
	if chunked {
		w.header.Set("Transfer-Encoding", "chunked")
	} else if w.header.Get("Content-Length") == "" {
		w.header.Set("Content-Length", strconv.Itoa(w.buf.Len()))
	}
	w.header.Set("Connection", "close")

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("HTTP/1.1 %d %s\r\n", w.statusCode, http.StatusText(w.statusCode)))
	for k, vv := range w.header {
		for _, v := range vv {
			sb.WriteString(fmt.Sprintf("%s: %s\r\n", k, v))
		}
	}
	sb.WriteString("\r\n")
	_ = w.conn.SetWriteDeadline(time.Now().Add(mitmIdleDeadline))
	// An audit caught that the original tlsConn.SetDeadline call before
	// this handler started (serveInterceptedTLS) sets BOTH read and write
	// deadlines to the same absolute instant, but only the write side ever
	// gets refreshed (here and in writeNewBytes below) — the disconnect-
	// watcher goroutine's own tlsConn.Read is still bound by that original,
	// never-renewed READ deadline, so a genuinely long-but-active
	// generation gets its connection killed exactly mitmIdleDeadline after
	// it STARTED, not after it went idle. Refresh the read side on every
	// chunk too, the same idle-timeout pattern this file already uses
	// correctly elsewhere (serveTunneledConnection's src/dst deadline
	// refresh) — this keeps a real upper bound against a leaked goroutine
	// on a silently-dead connection, just one that resets on genuine
	// activity instead of the connection's start time.
	_ = w.conn.SetReadDeadline(time.Now().Add(mitmIdleDeadline))
	_, _ = w.conn.Write([]byte(sb.String()))
}

// writeNewBytes pushes only what's been written to w.buf since the last
// call — not the whole buffer — as either a raw slice (fixed
// Content-Length framing) or one chunked-encoding frame. Previously Flush()
// replayed the status line, every header, and the ENTIRE buffer
// accumulated so far on every single call, so any stream with more than
// one chunk (every real SSE response) sent the client a response with its
// headers and earlier chunks duplicated over and over.
func (w *MitmResponseWriter) writeNewBytes() {
	all := w.buf.Bytes()
	if w.sentOffset >= len(all) {
		return
	}
	chunk := all[w.sentOffset:]
	w.sentOffset = len(all)
	_ = w.conn.SetWriteDeadline(time.Now().Add(mitmIdleDeadline))
	// Same read-deadline refresh as sendHeaders above, and for the same
	// reason — this is the per-chunk activity point for a real SSE stream.
	_ = w.conn.SetReadDeadline(time.Now().Add(mitmIdleDeadline))
	if w.chunked {
		_, _ = w.conn.Write([]byte(fmt.Sprintf("%x\r\n", len(chunk))))
		_, _ = w.conn.Write(chunk)
		_, _ = w.conn.Write([]byte("\r\n"))
	} else {
		_, _ = w.conn.Write(chunk)
	}
}

// Flush is called mid-handler by anything holding this writer as an
// http.Flusher (SSEStreamer.SendRaw, in particular) — exactly the case
// sendHeaders(true)/writeNewBytes above exist to make safe.
func (w *MitmResponseWriter) Flush() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if !w.headersSent {
		w.sendHeaders(true)
	}
	w.writeNewBytes()
}

// Close finalizes the response after the handler has returned. If Flush
// was never called mid-handler (every single-shot JSON handler — oauth/
// usage, bootstrap, models, non-streaming /v1/messages), this is the ONLY
// flush point and the full body is already buffered, so it's sent once
// with an exact Content-Length — unchanged from the old behavior. If Flush
// WAS already called (a real SSE stream), headers and earlier chunks are
// already on the wire in chunked framing; this just sends the trailing
// bytes and the terminating zero-length chunk.
func (w *MitmResponseWriter) Close() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if !w.headersSent {
		w.sendHeaders(false)
	}
	w.writeNewBytes()
	if w.chunked {
		_ = w.conn.SetWriteDeadline(time.Now().Add(mitmIdleDeadline))
		_, _ = w.conn.Write([]byte("0\r\n\r\n"))
	}
}
