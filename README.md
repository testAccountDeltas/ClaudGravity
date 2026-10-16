# ClaudGravity

**Claude Code, but the backend is Google's free Antigravity/Gemini pool — and it feels native.**

ClaudGravity — высокопроизводительный Go-шлюз, который транслирует запросы Claude Code (Anthropic Messages API) в Google CloudCode / Antigravity internal API и Gemini API, с пулом Google-аккаунтов, кэшем сессий и полной эмуляцией нативного клиентского опыта. Claude Code подключается к нему как к `https://api.anthropic.com` — и не замечает разницы.

> Работает **только** с Claude Code и **только** через Antigravity/Gemini-бэкенд. Это не универсальный LLM-роутер — узкий, целевой мост под одну конкретную пару клиент↔бэкенд, доведённый до мелочей.

---

## Почему "как родной" — не маркетинг, а список конкретных вещей

Большинство самодельных прокси для Claude Code ломаются на нюансах, которые не видны в happy path. ClaudGravity закрывает именно их:

- **Оба прогресс-бара `/usage` (5h + 7d) работают** — шлюз эмулирует `anthropic-ratelimit-unified-*` заголовки и перехватывает hardcoded `/api/oauth/usage`/`/api/claude_cli/bootstrap` (claude.exe жёстко стучится на `api.anthropic.com`, игнорируя `ANTHROPIC_BASE_URL`) через собственный MITM CA + CONNECT-прокси — без внешних mitmproxy/Python-процессов.
- **Полный стек субагентов** (`Agent`/`Task`/`PlanMode`/`TodoWrite`) не ломается — учтены все скрытые переменные окружения и заголовки, которые Claude Code использует, чтобы решить, доверять ли модели инструменты.
- **Prompt caching реально работает**: детерминированная Session Affinity привязывает диалог к одному Google-аккаунту по контекстному якорю — без этого Gemini теряет серверный префикс-кэш на каждом переключении аккаунта, и каждый ход стоит как первый.
- **Контекстное окно считается по реальной под-модели**, а не по статическому `CLAUDE_CODE_MAX_CONTEXT_TOKENS`: тиринг может на лету менять Gemini-подмодель с разным реальным лимитом (131K…1M), и усечение истории подстраивается под это в реальном времени, с резервом под инфляцию `thinking_budget`.
- **Переживает реальный Google-инфраструктурный потолок** на ~1,048,576 токенов сессионного кэша: при попадании в стену — детерминированная ротация `sessionId` и повтор того же запроса, без потери части диалога и без слепого переключения на другой аккаунт.
- **Fallback по географии бэкенда**: если конкретный CloudCode-хост отбивает по location-gate, шлюз пробует следующий хост с тем же запросом — вместо фейла всей сессии.
- **Chrome TLS ClientHello (uTLS)** на исходящих вызовах к Google — не торчит голым Go-stdlib отпечатком.
- **LoopGuard**: детект read-циклов и осцилляций инструментов, мягкий нудж вместо зависания агента.
- **Самовосстанавливающийся граф сообщений**: склейка однородных ролей, лечение осиротевших `tool_result`, восстановительные заглушки для брошенных `tool_use` — типичные причины 400 от бэкендов без AST-санитайзера.
- **Нативный веб-поиск**: `WebSearch` исполняется через Google Search Grounding напрямую, не гоняя лишний раунд через тяжёлую модель.
- **Авто-починка `Edit`/`Update`/`NotebookEdit` при расхождении табов/пробелов/CRLF** — и это работает даже когда шлюз развёрнут на отдельном сервере без доступа к диску клиента: ground truth берётся не с диска, а из истории диалога (последний `Read`/`Write`/`Edit` по этому пути в самом запросе) — то есть чинится именно то расхождение, которое модель сама внесла, транскрибируя собственный контекст.
- **Веб-дашборд** с живым WebSocket-потоком, деревом сессии (включая параллельные инстансы одного сабагента — различаются по `X-Claude-Code-Prompt-Id`), управлением пулом аккаунтов, фильтрами.

---

## Архитектура

```
Claude Code (claude.exe)
   │
   ├─ HTTPS_PROXY ──────► CONNECT-прокси (:8898) ─┬─ /api/oauth/usage      (synth.)
   │                                                ├─ /api/claude_cli/*    (synth.)
   │                                                └─ * (честный passthrough)
   │
   └─ ANTHROPIC_BASE_URL ► TLS API (:8900) /v1/messages
                               │
                               ▼
                 Translator → Sanitizer → LoopGuard → Context-window truncation
                               │
                               ▼
                     Google CloudCode / Antigravity / Gemini API
                     (пул OAuth-аккаунтов, session affinity, тиринг)
```

Оба порта (`:8898` CONNECT-прокси, `:8900` TLS API) поднимаются одним процессом и используют один и тот же самоподписанный CA (`certs/superrelay-ca.pem`), который клиент должен один раз довериться (`SSL_CERT_FILE`/`NODE_EXTRA_CA_CERTS`).

---

## Быстрый старт

### Локально (шлюз и Claude Code на одной машине)

```bash
go build -o gateway.exe .
cp config.example.json config.json   # заполнить google_api_key или настроить auths/ (OAuth-пул)
./gateway.exe config.json
```

Переменные окружения для Claude Code (пример для Windows, см. `claude-go-relay.cmd`):

```
HTTPS_PROXY=http://<proxy_secret>:x@127.0.0.1:8898
ANTHROPIC_BASE_URL=https://127.0.0.1:8900
SSL_CERT_FILE=<repo>/certs/superrelay-ca.pem
NODE_EXTRA_CA_CERTS=<repo>/certs/superrelay-ca.pem
```

`proxy_secret`/`dashboard_password` генерируются автоматически при первом старте и пишутся в `config.json` + лог.

### На удалённом сервере

Шлюзу не нужен доступ к дискам клиента — всё, что ему нужно знать о файлах, оно уже получает из истории диалога. Единственное, что важно настроить — `server.public_host` в конфиге (IP/домен, по которому клиент реально будет подключаться): он попадает в SAN TLS-сертификата, иначе проверка hostname на клиенте упадёт.

```json
{
  "server": {
    "tls_port": 8900,
    "public_host": "203.0.113.10"
  }
}
```

Деплой как systemd-юнит, пример — в `SUPERRELAY-SERVICE.md`.

---

## Конфигурация

См. `config.example.json`. Ключевые поля `server`:

| Поле | Назначение |
|---|---|
| `port` / `tls_port` | CONNECT-прокси / TLS API для `ANTHROPIC_BASE_URL` |
| `public_host` | hostname/IP, под который минтится TLS-сертификат (SAN) — нужен при удалённом деплое |
| `panic_timeout_seconds` | через сколько секунд молчания от бэкенда шлюз шлёт ранний SSE-keepalive, чтобы клиент не счёл запрос мёртвым |
| `ttft_timeout_seconds` / `read_timeout_seconds` | тайм-ауты до первого байта / на всё чтение |
| `dashboard_password` / `proxy_secret` | авто-генерируются при первом запуске |

Ключевые поля `tenants[key]`: `google_auth_type` (`api_key` или OAuth-пул из `auths/`), `default_model`, `thinking_budget`, `strip_search_tools`, `sticky_account`, `strategy`.

---

## Безопасность

- CONNECT-прокси требует `Proxy-Authorization` (`proxy_secret`) — иначе это был бы открытый релей.
- Дашборд закрыт Basic Auth (`admin` / `dashboard_password` из лога при старте).
- Приватный ключ CA (`certs/superrelay-ca.key`) и реальные OAuth-токены аккаунтов (`auths/*.json`) — **никогда** не коммитятся (см. `.gitignore`). Если разворачиваешь на чужом сервере — это чувствительные данные, передавай отдельно.

---

## Ограничения

- Узкоспециализирован под Claude Code↔Antigravity/Gemini; не универсальный LLM-прокси.
- Авто-починка `Edit` по снапшоту диалога чинит расхождения между тем, что модель сама читала/писала в рамках текущего запроса — она не видит правок файла, сделанных человеком мимо модели между вызовами, если шлюз не имеет локального доступа к диску.
- Если шлюз и клиент на разных машинах, качество соединения зависит от сети между ними — удалённый CONNECT-прокси и TLS API делят один и тот же инстанс, и при точках назначения на одном сервере трафик идёт через hairpin по публичному IP, а не локальной петлёй.

---

## Благодарности

Несколько эвристик восстановления (ротация `sessionId` на 1M-стене, host-fallback по location-gate) независимо найдены и в другом проекте — [clirelay](https://github.com/kittors/CliRelay); где это так, это отмечено в комментариях кода.

## Лицензия

MIT, см. `LICENSE`.
