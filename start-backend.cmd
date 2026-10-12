@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-backend.ps1"
if errorlevel 1 (
    echo.
    echo CarWorld backend startup failed. Review the error above.
    pause
)
