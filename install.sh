#!/usr/bin/env bash
# ClaudGravity installer for macOS/Linux.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/testAccountDeltas/ClaudGravity/main/install.sh | bash
#
# Non-interactive:
#   CLAUDGRAVITY_URL="1.2.3.4:8898" CLAUDGRAVITY_KEY="sk-..." \
#     curl -fsSL https://raw.githubusercontent.com/testAccountDeltas/ClaudGravity/main/install.sh | bash
#
# Uninstall:
#   curl -fsSL https://raw.githubusercontent.com/testAccountDeltas/ClaudGravity/main/install.sh | bash -s -- --uninstall

set -euo pipefail

INSTALL_DIR="$HOME/.claudgravity"
OS="$(uname -s)"

say()  { printf "\033[36m==> %s\033[0m\n" "$1"; }
ok()   { printf "    \033[32m%s\033[0m\n" "$1"; }
warn() { printf "    \033[33m%s\033[0m\n" "$1"; }

if [ "${1:-}" = "--uninstall" ]; then
  say "Удаляю ClaudGravity..."
  rm -rf "$INSTALL_DIR"
  if [ -f "$HOME/.local/share/nautilus/scripts/Запустить Claude Code тут" ]; then
    rm -f "$HOME/.local/share/nautilus/scripts/Запустить Claude Code тут"
  fi
  rm -f "$HOME/Desktop/ClaudGravity.command" 2>/dev/null || true
  for rc in "$HOME/.zshrc" "$HOME/.bashrc"; do
    [ -f "$rc" ] && sed -i.bak "/# >>> claudgravity >>>/,/# <<< claudgravity <<</d" "$rc" 2>/dev/null || true
  done
  ok "Готово."
  exit 0
fi

echo ""
echo "  ClaudGravity installer"
echo "  Claude Code <-> Google Antigravity/Gemini gateway client setup"
echo ""

# ───────────────────────── 1. gateway url + key ─────────────────────────
say "Параметры шлюза"

GATEWAY_URL="${CLAUDGRAVITY_URL:-}"
if [ -z "$GATEWAY_URL" ]; then
  read -rp "Адрес шлюза (например 203.0.113.10 или 203.0.113.10:8898): " GATEWAY_URL
fi
GATEWAY_URL="${GATEWAY_URL%/}"
GATEWAY_URL="${GATEWAY_URL#http://}"
GATEWAY_URL="${GATEWAY_URL#https://}"

API_KEY="${CLAUDGRAVITY_KEY:-}"
if [ -z "$API_KEY" ]; then
  read -rp "API-ключ (sk-...): " API_KEY
fi

if [[ "$GATEWAY_URL" != *:* ]]; then
  GATEWAY_URL="$GATEWAY_URL:8898"
fi
CONNECT_HOST="${GATEWAY_URL%%:*}"
CONNECT_PORT="${GATEWAY_URL##*:}"

say "Получаю настройки с шлюза ($CONNECT_HOST:$CONNECT_PORT)..."
BOOTSTRAP_JSON="$(curl -fsS -m 10 "http://$CONNECT_HOST:$CONNECT_PORT/api/bootstrap?key=$API_KEY" || true)"
if [ -z "$BOOTSTRAP_JSON" ]; then
  echo "Не удалось связаться со шлюзом или ключ неверный." >&2
  exit 1
fi

json_get() { printf '%s' "$BOOTSTRAP_JSON" | sed -n "s/.*\"$1\":\"\\?\\([^\",}]*\\).*/\\1/p" | head -1; }
# ca_pem contains \n escapes and embedded quotes-free PEM text; extract separately.
PUBLIC_HOST="$(json_get public_host)"
TLS_PORT="$(json_get tls_port)"
PROXY_SECRET="$(json_get proxy_secret)"
CA_PEM="$(printf '%s' "$BOOTSTRAP_JSON" | sed -n 's/.*"ca_pem":"\([^"]*\)".*/\1/p')"

if [ -z "$CA_PEM" ] || [ -z "$TLS_PORT" ]; then
  echo "Шлюз ответил, но не выдал CA-сертификат/tls_port — проверь версию шлюза." >&2
  echo "Raw response: $BOOTSTRAP_JSON" >&2
  exit 1
fi
if [ -z "$PUBLIC_HOST" ] || [ "$PUBLIC_HOST" = "localhost" ]; then
  PUBLIC_HOST="$CONNECT_HOST"
fi
ok "OK — tls_port=$TLS_PORT public_host=$PUBLIC_HOST"

# ───────────────────────── 2. save local files ─────────────────────────
say "Сохраняю конфигурацию в $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
# %b un-escapes the \n sequences the JSON encoding left literal.
printf '%b' "$CA_PEM" > "$INSTALL_DIR/ca.pem"
cat > "$INSTALL_DIR/config.json" <<EOF
{
  "gateway_host": "$CONNECT_HOST",
  "connect_port": "$CONNECT_PORT",
  "tls_port": "$TLS_PORT",
  "public_host": "$PUBLIC_HOST",
  "proxy_secret": "$PROXY_SECRET",
  "api_key": "$API_KEY"
}
EOF
ok "ca.pem + config.json сохранены"

# ───────────────────────── 3. find claude ─────────────────────────
# Порядок: VS Code extension -> PATH/переменные среды -> нативные пути установки.
say "Ищу установку Claude Code..."
CLAUDE_CMD=""
FOUND_VIA=""

find_in_vscode() {
  local ext_glob
  if [ "$OS" = "Darwin" ]; then
    ext_glob="$HOME/.vscode/extensions/anthropic.claude-code-*-darwin-"*
  else
    ext_glob="$HOME/.vscode/extensions/anthropic.claude-code-*-linux-"*
  fi
  for d in $ext_glob; do
    [ -d "$d" ] || continue
    for bin in "$d/resources/native-binary/claude" "$d/resources/native-binary/claude-code"; do
      [ -f "$bin" ] && { echo "$bin"; return 0; }
    done
  done
  return 1
}

find_in_path() {
  command -v claude 2>/dev/null && return 0
  if [ -n "${CLAUDGRAVITY_CLAUDE_PATH:-}" ] && [ -x "${CLAUDGRAVITY_CLAUDE_PATH:-}" ]; then
    echo "$CLAUDGRAVITY_CLAUDE_PATH"; return 0
  fi
  return 1
}

find_native() {
  for c in \
    "$HOME/.local/bin/claude" \
    "/usr/local/bin/claude" \
    "/opt/homebrew/bin/claude"
  do
    [ -x "$c" ] && { echo "$c"; return 0; }
  done
  if command -v npm >/dev/null 2>&1; then
    local npm_bin
    npm_bin="$(npm bin -g 2>/dev/null || true)"
    if [ -n "$npm_bin" ] && [ -x "$npm_bin/claude" ]; then
      echo "$npm_bin/claude"; return 0
    fi
  fi
  return 1
}

if out="$(find_in_vscode)"; then CLAUDE_CMD="$out"; FOUND_VIA="VS Code extension"; fi
if [ -z "$CLAUDE_CMD" ] && out="$(find_in_path)"; then CLAUDE_CMD="$out"; FOUND_VIA="PATH/переменные среды"; fi
if [ -z "$CLAUDE_CMD" ] && out="$(find_native)"; then CLAUDE_CMD="$out"; FOUND_VIA="нативная установка"; fi

if [ -n "$CLAUDE_CMD" ]; then
  ok "Найден ($FOUND_VIA): $CLAUDE_CMD"
else
  warn "Claude Code не найден автоматически. launch.sh будет использовать 'claude' из PATH."
  CLAUDE_CMD="claude"
  FOUND_VIA="не найден автоматически, используется 'claude' из PATH"
fi

# ───────────────────────── 4. write launch.sh ─────────────────────────
say "Создаю launch.sh"
LAUNCH_PATH="$INSTALL_DIR/launch.sh"
cat > "$LAUNCH_PATH" <<EOF
#!/usr/bin/env bash
export CLAUDE_CONFIG_DIR="$INSTALL_DIR/profile"
export ANTHROPIC_BASE_URL="https://$PUBLIC_HOST:$TLS_PORT"
export HTTPS_PROXY="http://$PROXY_SECRET:x@$CONNECT_HOST:$CONNECT_PORT"
export SSL_CERT_FILE="$INSTALL_DIR/ca.pem"
export NODE_EXTRA_CA_CERTS="$INSTALL_DIR/ca.pem"
unset HTTP_PROXY ALL_PROXY ANTHROPIC_AUTH_TOKEN ANTHROPIC_API_KEY
export CLAUDE_CODE_OAUTH_TOKEN="$API_KEY"
export CLAUDE_CODE_OAUTH_SCOPES="user:inference user:profile"
export CLAUDE_CODE_SUBSCRIPTION_TYPE="max"
export CLAUDE_CODE_RATE_LIMIT_TIER="default"
export CLAUDE_CODE_ENABLE_TODO_TOOLS=1
export CLAUDE_CODE_ATTRIBUTION_HEADER=0
export DISABLE_AUTOUPDATER=1
echo "ClaudGravity -> $CONNECT_HOST:$CONNECT_PORT ($FOUND_VIA)"
exec "$CLAUDE_CMD" "\$@"
EOF
chmod +x "$LAUNCH_PATH"
ok "launch.sh готов"

# ───────────────────────── 5. shell integration (claudgravity command) ─────────────────────────
say "Добавляю команду 'claudgravity' в shell"
BLOCK=$'\n# >>> claudgravity >>>\nclaudgravity() { "'"$LAUNCH_PATH"'" "$@"; }\n# <<< claudgravity <<<\n'
for rc in "$HOME/.zshrc" "$HOME/.bashrc"; do
  if [ -f "$rc" ]; then
    if ! grep -q "# >>> claudgravity >>>" "$rc"; then
      printf '%s' "$BLOCK" >> "$rc"
      ok "Добавлено в $rc"
    fi
  fi
done

# ───────────────────────── 6. OS-specific "open here" integration ─────────────────────────
if [ "$OS" = "Linux" ]; then
  say "Устанавливаю Nautilus-скрипт (правый клик -> Scripts)"
  NAUT_DIR="$HOME/.local/share/nautilus/scripts"
  mkdir -p "$NAUT_DIR"
  NAUT_SCRIPT="$NAUT_DIR/Запустить Claude Code тут"
  cat > "$NAUT_SCRIPT" <<EOF
#!/usr/bin/env bash
cd "\$NAUTILUS_SCRIPT_CURRENT_URI" 2>/dev/null || cd "\$PWD"
x-terminal-emulator -e "$LAUNCH_PATH" 2>/dev/null || gnome-terminal -- "$LAUNCH_PATH" 2>/dev/null || "$LAUNCH_PATH"
EOF
  chmod +x "$NAUT_SCRIPT"
  ok "Готово (если файловый менеджер не Nautilus — используй команду 'claudgravity' в терминале из нужной папки)"
elif [ "$OS" = "Darwin" ]; then
  say "Создаю ClaudGravity.command на Desktop (можно перетащить в Dock)"
  DESKTOP_CMD="$HOME/Desktop/ClaudGravity.command"
  if [ -d "$HOME/Desktop" ]; then
    cat > "$DESKTOP_CMD" <<EOF
#!/usr/bin/env bash
cd "\$HOME"
"$LAUNCH_PATH"
EOF
    chmod +x "$DESKTOP_CMD"
    ok "Создано: $DESKTOP_CMD"
  fi
  warn "Полноценный пункт в контекстном меню Finder требует Automator Quick Action — используй команду 'claudgravity' в терминале из нужной папки."
fi

echo ""
echo "  Готово!"
echo "  Открой НОВЫЙ терминал и запусти: claudgravity"
echo "  (Linux: также правый клик в Nautilus -> Scripts -> \"Запустить Claude Code тут\")"
echo "  Удалить: curl -fsSL <this-script-url> | bash -s -- --uninstall"
echo ""
