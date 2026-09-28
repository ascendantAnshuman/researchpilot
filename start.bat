@echo off
rem Double-click to start ResearchPilot (runs start.ps1 without changing PowerShell's execution policy).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
if errorlevel 1 pause
