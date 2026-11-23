@echo off
cd /d "%~dp0"
.venv\Scripts\python ninja_cam.py %*
pause
