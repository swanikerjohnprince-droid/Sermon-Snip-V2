@echo off
setlocal EnableExtensions
cd /d "%~dp0"
rem Installs verthor (KazKozDev/auto-vertical-reframe, MIT) into its own environment: .venv-verthor
rem Pinned to the commit Sermon-Snip was reviewed against. Needs Python 3.11 or 3.12 (mediapipe 0.10.14
rem has no Python 3.13 wheels). Download is large (PyTorch + YOLO + MediaPipe, a few GB).

set "PYCMD=python"
where py >nul 2>nul
if not errorlevel 1 (
  py -3.12 -c "import sys" >nul 2>nul && set "PYCMD=py -3.12"
  if "%PYCMD%"=="python" py -3.11 -c "import sys" >nul 2>nul && set "PYCMD=py -3.11"
)
%PYCMD% -c "import sys; sys.exit(0 if (3,11)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>nul
if errorlevel 1 (
  echo verthor needs Python 3.11 or 3.12 ^(64-bit^). Install one from python.org and run this again.
  pause
  exit /b 1
)

if not exist ".venv-verthor\Scripts\python.exe" (
  echo Creating the verthor environment...
  %PYCMD% -m venv .venv-verthor
  if errorlevel 1 goto :fail
)
".venv-verthor\Scripts\python.exe" -m pip install --upgrade pip
".venv-verthor\Scripts\python.exe" -m pip install "https://github.com/KazKozDev/auto-vertical-reframe/archive/1556b3c98b07379af196b1de0dc921c36663269a.zip"
if errorlevel 1 goto :fail

echo.
echo verthor installed.
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
  echo.
  echo An NVIDIA GPU was found. PyPI's Windows PyTorch is CPU-only, so verthor would be slow.
  echo Installing the CUDA build of PyTorch for your driver ^(about 2.5 GB, one time^)...
  if exist ".venv\Scripts\python.exe" ( ".venv\Scripts\python.exe" verthor_gpu.py --install ) else ( python verthor_gpu.py --install )
  if errorlevel 1 echo GPU setup did not finish. You can retry it from the Reframe step ^("Enable GPU"^) or run this file again.
) else (
  echo No NVIDIA GPU detected, so verthor will run on the CPU.
)
echo Sermon-Snip finds verthor automatically ^(or use "Locate tools..."^).
pause
exit /b 0

:fail
echo.
echo Install failed. See the messages above.
pause
exit /b 1
