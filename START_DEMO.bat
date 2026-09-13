@echo off
setlocal enabledelayedexpansion
title Marinedetect
cd /d "%~dp0"

echo.
echo   Marinedetect
echo   Side-scan sonar detection and shadow-geometry measurement
echo   ---------------------------------------------------------
echo.

:: --------------------------------------------------------------------
:: 1. Find a usable Python. Note the pip check: a Python without pip is
::    worse than no Python, because it fails later and less clearly.
:: --------------------------------------------------------------------
set "PY="
if exist ".venv\Scripts\python.exe" (
    set "PY=%~dp0.venv\Scripts\python.exe"
    goto :have_env
)

:: Prefer a version torch supports. A bare "py -3" can land on 3.14, which
:: has no torch wheels; setup.py guards against that too, but be explicit.
for %%v in (3.12 3.13 3.11 3.10) do (
    py -%%v --version >nul 2>&1
    if !errorlevel! equ 0 ( set "PY=py -%%v" & goto :setup )
)
py -3 --version >nul 2>&1
if !errorlevel! equ 0 ( set "PY=py -3" & goto :setup )

python --version >nul 2>&1
if !errorlevel! equ 0 (
    python -m pip --version >nul 2>&1
    if !errorlevel! equ 0 ( set "PY=python" & goto :setup )
)

echo   Python was not found.
echo.
echo   Install Python 3.12 from https://www.python.org/downloads/
echo   During installation, tick "Add python.exe to PATH".
echo.
pause
exit /b 1

:: --------------------------------------------------------------------
:: 2. First run only. setup.py detects whether this machine has an NVIDIA
::    GPU and installs the matching PyTorch build, then verifies it.
::    Takes 5-15 minutes, mostly downloading PyTorch.
:: --------------------------------------------------------------------
:setup
echo   First-time setup. This runs once and takes 5-15 minutes.
echo   Most of that is downloading PyTorch - leave it be.
echo.
%PY% scripts\setup.py
if !errorlevel! neq 0 (
    echo.
    echo   Setup did not complete. The messages above say what failed.
    pause
    exit /b 1
)
set "PY=%~dp0.venv\Scripts\python.exe"

:have_env

:: --------------------------------------------------------------------
:: 3. Free port 8000 if something is already sitting on it.
:: --------------------------------------------------------------------
for /f "tokens=5" %%a in ('netstat -ano ^| findstr "LISTENING" ^| findstr ":8000 " 2^>nul') do set "PID=%%a"
if defined PID (
    echo   Port 8000 was busy - stopping the old process.
    taskkill /PID !PID! /F >nul 2>&1
    ping -n 3 127.0.0.1 >nul 2>&1
)

:: --------------------------------------------------------------------
:: 4. Start the server and open the browser.
:: --------------------------------------------------------------------
echo.
echo   Starting. The browser will open at http://127.0.0.1:8000
echo   Leave this window open. Close it, or press Ctrl+C, to stop.
echo.
start "" /b powershell -WindowStyle Hidden -Command "Start-Sleep -Seconds 6; Start-Process 'http://127.0.0.1:8000'"
"%PY%" -m uvicorn backend.main:app --app-dir "%~dp0." --host 127.0.0.1 --port 8000

echo.
echo   Server stopped.
pause
