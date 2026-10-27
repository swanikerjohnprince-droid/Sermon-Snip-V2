@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run INSTALL_AND_RUN.bat first to create the environment.
  pause
  exit /b 1
)
echo Installing the NVIDIA CUDA 12 runtime libraries ^(about 1.3 GB^) into this app's environment.
echo Your system CUDA / drivers are not modified.
".venv\Scripts\python.exe" -m pip install -r requirements-gpu.txt
if errorlevel 1 (
  echo.
  echo Install failed. See the messages above.
  pause
  exit /b 1
)
copy /y requirements-gpu.txt ".venv\installed_requirements-gpu.txt" >nul
echo.
".venv\Scripts\python.exe" gpu_setup.py --selftest
pause
