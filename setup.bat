@echo off
REM YDS - YOLO Dataset Studio Setup Script for Windows

echo.
echo ========================================
echo    YDS Setup - YOLO Dataset Studio
echo ========================================
echo.

set "VENV_DIR=.venv"

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/6] Python detected:
python --version

echo [2/6] Creating virtual environment...
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    python -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
) else (
    echo Using existing venv: %VENV_DIR%
)

echo [3/6] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

echo [4/6] Checking requirements.txt...
if not exist requirements.txt (
    echo ERROR: requirements.txt not found
    pause
    exit /b 1
)

echo [5/6] Installing dependencies...
python -m pip install --upgrade pip setuptools wheel >nul 2>&1
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo [6/6] Verifying installation...
python -c "from ultralytics import YOLO; print('[OK] YOLO installed')" >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Installation verification failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo    Setup Complete!
echo ========================================
echo.
echo To launch the application, run:
echo    python GUI.py
echo.
pause
