@echo off
setlocal
cd /d "%~dp0"

set "FRONTEND_ENV=development"
if not defined FRONTEND_API_BASE_URL set "FRONTEND_API_BASE_URL=http://127.0.0.1:8000"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0backend\scripts\render-frontend-config.ps1"
if errorlevel 1 (
    echo Failed to configure the frontend API URL.
    pause
    exit /b 1
)

set "PYTHON=%~dp0.venv-1\Scripts\python.exe"
if not exist "%PYTHON%" (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python was not found. Install Python or create the project's .venv-1 environment.
        pause
        exit /b 1
    )
    set "PYTHON=python"
)

echo Serving CarWorld frontend at http://127.0.0.1:5500/index.html
echo Press Ctrl+C to stop the frontend server.
"%PYTHON%" -m http.server 5500 --bind 127.0.0.1
if errorlevel 1 (
    echo.
    echo Frontend server stopped with an error. Check whether port 5500 is already in use.
    pause
)
