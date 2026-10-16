@echo off
setlocal

cd /d "%~dp0"

tasklist /FI "IMAGENAME eq gateway.exe" 2>NUL | find /I "gateway.exe" >NUL
if %ERRORLEVEL%==0 (
    echo Stopping existing gateway.exe...
    taskkill /IM gateway.exe /F >NUL 2>&1
    timeout /t 1 /nobreak >NUL
)

echo Starting SuperRelay Gateway...
start "SuperRelay Gateway" gateway.exe config.json

endlocal
