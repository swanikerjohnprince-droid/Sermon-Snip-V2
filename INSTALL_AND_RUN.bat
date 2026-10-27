@echo off
setlocal EnableExtensions
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install 64-bit Python 3.11 or 3.12 and tick "Add Python to PATH".
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating a private Python environment ^(first run only^)...
  python -m venv .venv
  if errorlevel 1 goto :fail
)
set "PY=.venv\Scripts\python.exe"

rem With an NVIDIA driver present, also install the pinned CUDA 12 runtime wheels (about 1.3 GB, once).
set "REQ=requirements.txt"
where nvidia-smi >nul 2>nul
if not errorlevel 1 set "REQ=requirements-gpu.txt"

fc /b "%REQ%" ".venv\installed_%REQ%" >nul 2>nul
if errorlevel 1 (
  echo Installing dependencies from %REQ% ...
  if "%REQ%"=="requirements-gpu.txt" echo ^(This includes about 1.3 GB of NVIDIA libraries; it only happens once.^)
  "%PY%" -m pip install --upgrade pip
  "%PY%" -m pip install -r "%REQ%"
  if errorlevel 1 goto :fail
  copy /y "%REQ%" ".venv\installed_%REQ%" >nul
)

"%PY%" -c "import media,sys; sys.exit(0 if media.ffmpeg_available() else 1)" >nul 2>nul
if errorlevel 1 (
  echo.
  echo FFmpeg was not found. Sermon-Snip needs it for previews, cutting, reframing and captions.
  choice /c YN /m "Install FFmpeg now (about 200 MB, no admin needed)"
  if not errorlevel 2 call "%~dp0INSTALL_FFMPEG.bat"
)

"%PY%" web_app.py
if errorlevel 1 pause
exit /b 0

:fail
echo.
echo Setup failed. See the messages above.
pause
exit /b 1
