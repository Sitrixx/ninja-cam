@echo off
cd /d "%~dp0"
set PY=
for %%v in (3.12 3.11 3.10) do (
  if not defined PY ( py -%%v -c "1" >nul 2>nul && set PY=py -%%v )
)
if not defined PY (
  echo Python 3.10, 3.11 or 3.12 is required. Get 3.12 from https://www.python.org/downloads/windows/
  pause & exit /b 1
)
echo Using %PY%
%PY% -m venv .venv || (pause & exit /b 1)
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt || (pause & exit /b 1)
echo.
echo Done! Install OBS Studio (https://obsproject.com) if you have not, then double-click run.bat
pause
