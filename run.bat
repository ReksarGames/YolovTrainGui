@echo off
REM YDS - YOLO Dataset Studio Run Script for Windows

set "VENV_DIR=.venv"

if not exist "%VENV_DIR%\\Scripts\\python.exe" (
    echo ERROR: .venv not found. Run setup.bat first.
    pause
    exit /b 1
)

call "%VENV_DIR%\\Scripts\\activate.bat"
python GUI.py
