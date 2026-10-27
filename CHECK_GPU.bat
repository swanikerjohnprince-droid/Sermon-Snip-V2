@echo off
setlocal EnableExtensions
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
echo Running the Sermon-Snip GPU diagnosis. The first run downloads a 75 MB test model.
echo.
"%PY%" gpu_setup.py --selftest > gpu_report.txt 2>&1
type gpu_report.txt
echo.
echo The report was saved to gpu_report.txt - send that file if you need help.
pause
