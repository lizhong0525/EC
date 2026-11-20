@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 goto usepython
py -3 -B server.py
if errorlevel 1 pause
exit /b
:usepython
python -B server.py
if errorlevel 1 pause
