#Requires -Version 5.1
<#
  ClaudGravity installer for Windows.

  Usage (one-liner):
    irm https://raw.githubusercontent.com/testAccountDeltas/ClaudGravity/main/install.ps1 | iex

  Non-interactive (CI / scripted):
    $env:CLAUDGRAVITY_URL = "http://1.2.3.4:8898"
    $env:CLAUDGRAVITY_KEY = "sk-..."
    irm https://raw.githubusercontent.com/testAccountDeltas/ClaudGravity/main/install.ps1 | iex

  Uninstall:
    irm https://raw.githubusercontent.com/testAccountDeltas/ClaudGravity/main/install.ps1 -OutFile install.ps1
    .\install.ps1 -Uninstall
#>
param(
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"
$InstallDir = Join-Path $env:LOCALAPPDATA "ClaudGravity"
$MenuLabel = "ClaudGravity: Запустить Claude Code тут"

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "    $msg" -ForegroundColor Yellow }

# ───────────────────────── uninstall ─────────────────────────
if ($Uninstall) {
    Write-Step "Удаляю ClaudGravity..."
    foreach ($base in @(
        "Registry::HKEY_CURRENT_USER\Software\Classes\Directory\Background\shell\ClaudGravity",
        "Registry::HKEY_CURRENT_USER\Software\Classes\Directory\shell\ClaudGravity"
    )) {
        if (Test-Path $base) { Remove-Item -Path $base -Recurse -Force -ErrorAction SilentlyContinue }
    }
    if (Test-Path $InstallDir) { Remove-Item -Path $InstallDir -Recurse -Force -ErrorAction SilentlyContinue }
    Write-Ok "Готово. Контекстное меню и $InstallDir удалены."
    return
}

Write-Host ""
Write-Host "  ClaudGravity installer" -ForegroundColor Magenta
Write-Host "  Claude Code <-> Google Antigravity/Gemini gateway client setup"
Write-Host ""

# ───────────────────────── 1. gateway url + key ─────────────────────────
Write-Step "Параметры шлюза"

$GatewayUrl = $env:CLAUDGRAVITY_URL
if (-not $GatewayUrl) {
    $GatewayUrl = Read-Host "Адрес шлюза (например 72.56.108.240 или 72.56.108.240:8898)"
}
$GatewayUrl = $GatewayUrl.Trim().TrimEnd('/')
$GatewayUrl = $GatewayUrl -replace '^https?://', ''

$ApiKey = $env:CLAUDGRAVITY_KEY
if (-not $ApiKey) {
    $ApiKey = Read-Host "API-ключ (sk-...)"
}
$ApiKey = $ApiKey.Trim()

if ($GatewayUrl -notmatch ':\d+$') {
    $GatewayUrl = "$GatewayUrl`:8898"
}
$ConnectHost, $ConnectPort = $GatewayUrl -split ':', 2

Write-Step "Получаю настройки с шлюза ($ConnectHost`:$ConnectPort)..."
try {
    $bootstrap = Invoke-RestMethod -Uri "http://$ConnectHost`:$ConnectPort/api/bootstrap?key=$ApiKey" -Method Get -TimeoutSec 10
} catch {
    Write-Host "Не удалось связаться со шлюзом или ключ неверный: $_" -ForegroundColor Red
    exit 1
}
if (-not $bootstrap.ca_pem) {
    Write-Host "Шлюз ответил, но не выдал CA-сертификат — проверь версию шлюза." -ForegroundColor Red
    exit 1
}

$PublicHost = $bootstrap.public_host
if (-not $PublicHost -or $PublicHost -eq "localhost") { $PublicHost = $ConnectHost }
$TlsPort = $bootstrap.tls_port
$ProxySecret = $bootstrap.proxy_secret
Write-Ok "OK — tls_port=$TlsPort public_host=$PublicHost"

# ───────────────────────── 2. save local files ─────────────────────────
Write-Step "Сохраняю конфигурацию в $InstallDir"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$CaPath = Join-Path $InstallDir "ca.pem"
Set-Content -Path $CaPath -Value $bootstrap.ca_pem -Encoding ascii -NoNewline

$configObj = @{
    gateway_host = $ConnectHost
    connect_port = $ConnectPort
    tls_port     = $TlsPort
    public_host  = $PublicHost
    proxy_secret = $ProxySecret
    api_key      = $ApiKey
}
$configObj | ConvertTo-Json | Set-Content -Path (Join-Path $InstallDir "config.json") -Encoding utf8
Write-Ok "ca.pem + config.json сохранены"

# ───────────────────────── 3. find claude ─────────────────────────
# Порядок поиска осознанно такой: VS Code extension (самый частый способ
# поставить Claude Code) -> PATH/переменные среды -> нативные пути
# установки по умолчанию для каждой ОС.
Write-Step "Ищу установку Claude Code..."
$ClaudeCmd = $null

function Find-InVSCodeExtensions {
    $extRoot = Join-Path $env:USERPROFILE ".vscode\extensions"
    if (-not (Test-Path $extRoot)) { return $null }
    $dirs = Get-ChildItem -Path $extRoot -Directory -Filter "anthropic.claude-code-*-win32-x64" -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending
    foreach ($d in $dirs) {
        $candidate = Join-Path $d.FullName "resources\native-binary\claude.exe"
        if (Test-Path $candidate) { return $candidate }
    }
    return $null
}

function Find-InPathOrEnv {
    $cmd = Get-Command claude -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    if ($env:CLAUDGRAVITY_CLAUDE_PATH -and (Test-Path $env:CLAUDGRAVITY_CLAUDE_PATH)) {
        return $env:CLAUDGRAVITY_CLAUDE_PATH
    }
    return $null
}

function Find-NativeInstall {
    $candidates = @(
        (Join-Path $env:USERPROFILE ".local\bin\claude.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\claude-code\claude.exe"),
        (Join-Path $env:APPDATA "npm\claude.cmd"),
        (Join-Path $env:APPDATA "npm\claude.exe")
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    # npm global root, if npm is on PATH
    $npmCmd = Get-Command npm -ErrorAction SilentlyContinue
    if ($npmCmd) {
        try {
            $npmRoot = (& npm root -g 2>$null).Trim()
            if ($npmRoot) {
                foreach ($name in @("claude.cmd", "claude.exe", "claude")) {
                    $c = Join-Path $npmRoot "..\$name"
                    if (Test-Path $c) { return (Resolve-Path $c).Path }
                }
            }
        } catch {}
    }
    return $null
}

$ClaudeCmd = Find-InVSCodeExtensions
$foundVia = "VS Code extension"
if (-not $ClaudeCmd) { $ClaudeCmd = Find-InPathOrEnv; $foundVia = "PATH/переменные среды" }
if (-not $ClaudeCmd) { $ClaudeCmd = Find-NativeInstall; $foundVia = "нативная установка" }

if ($ClaudeCmd) {
    Write-Ok "Найден ($foundVia): $ClaudeCmd"
} else {
    Write-Warn "Claude Code не найден автоматически. launch.cmd будет использовать команду 'claude' из PATH — поставь Claude Code или пропиши путь вручную в $InstallDir\launch.cmd"
    $ClaudeCmd = "claude"
    $foundVia = "не найден автоматически, используется 'claude' из PATH"
}

# ───────────────────────── 4. detect Windows Terminal ─────────────────────────
$WtPath = (Get-Command wt.exe -ErrorAction SilentlyContinue).Source
if ($WtPath) {
    Write-Ok "Windows Terminal найден — будет использоваться для запуска"
} else {
    Write-Warn "Windows Terminal не найден — контекстное меню будет открывать обычный cmd.exe"
}

# ───────────────────────── 5. write launch.cmd ─────────────────────────
Write-Step "Создаю launch.cmd"
$LaunchPath = Join-Path $InstallDir "launch.cmd"
$launchContent = @"
@echo off
setlocal

set "CLAUDE_CONFIG_DIR=$InstallDir\profile"
set "ANTHROPIC_BASE_URL=https://$PublicHost`:$TlsPort"
set "HTTPS_PROXY=http://$ProxySecret`:x@$ConnectHost`:$ConnectPort"
set "SSL_CERT_FILE=$CaPath"
set "NODE_EXTRA_CA_CERTS=$CaPath"
set "HTTP_PROXY="
set "ALL_PROXY="
set "ANTHROPIC_AUTH_TOKEN="
set "ANTHROPIC_API_KEY="
set "CLAUDE_CODE_OAUTH_TOKEN=$ApiKey"
set "CLAUDE_CODE_OAUTH_SCOPES=user:inference user:profile"
set "CLAUDE_CODE_SUBSCRIPTION_TYPE=max"
set "CLAUDE_CODE_RATE_LIMIT_TIER=default"
set "CLAUDE_CODE_SIMPLE="
set "CLAUDE_CODE_ENABLE_TODO_TOOLS=1"
set "CLAUDE_CODE_USE_POWERSHELL_TOOL=1"
set "CLAUDE_CODE_ATTRIBUTION_HEADER=0"
set "DISABLE_AUTOUPDATER=1"

echo ClaudGravity -^> $ConnectHost`:$ConnectPort ($foundVia)
"$ClaudeCmd" %*
endlocal
"@
Set-Content -Path $LaunchPath -Value $launchContent -Encoding ascii
Write-Ok "launch.cmd готов"

# ───────────────────────── 6. context menu ─────────────────────────
Write-Step "Регистрирую контекстное меню"

if ($WtPath) {
    $menuCommandBg  = '"' + $WtPath + '" -d "%V" cmd /k "' + $LaunchPath + '"'
    $menuCommandDir = '"' + $WtPath + '" -d "%1" cmd /k "' + $LaunchPath + '"'
} else {
    $menuCommandBg  = 'cmd.exe /k "cd /d "%V" && "' + $LaunchPath + '""'
    $menuCommandDir = 'cmd.exe /k "cd /d "%1" && "' + $LaunchPath + '""'
}

function Set-ContextMenuEntry($baseKey, $command) {
    New-Item -Path $baseKey -Force | Out-Null
    Set-ItemProperty -Path $baseKey -Name "(Default)" -Value $MenuLabel
    Set-ItemProperty -Path $baseKey -Name "Icon" -Value "cmd.exe"
    $cmdKey = Join-Path $baseKey "command"
    New-Item -Path $cmdKey -Force | Out-Null
    Set-ItemProperty -Path $cmdKey -Name "(Default)" -Value $command
}

Set-ContextMenuEntry "Registry::HKEY_CURRENT_USER\Software\Classes\Directory\Background\shell\ClaudGravity" $menuCommandBg
Set-ContextMenuEntry "Registry::HKEY_CURRENT_USER\Software\Classes\Directory\shell\ClaudGravity" $menuCommandDir
Write-Ok "Пункт '$MenuLabel' добавлен в контекстное меню папок"

Write-Host ""
Write-Host "  Готово!" -ForegroundColor Magenta
Write-Host "  Правый клик в любой папке -> '$MenuLabel'"
Write-Host "  Или напрямую: $LaunchPath"
Write-Host "  Удалить:      irm <this-script-url> | iex -Uninstall  (или .\install.ps1 -Uninstall)"
Write-Host ""
