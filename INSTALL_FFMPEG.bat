@echo off
setlocal
cd /d "%~dp0"
echo Installing FFmpeg for Sermon-Snip...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_ffmpeg.ps1"
if errorlevel 1 (
  echo.
  echo FFmpeg could not be installed automatically. See the message above.
) else (
  echo.
  echo Done.
)
pause
