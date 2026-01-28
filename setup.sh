#!/usr/bin/env bash
set -euo pipefail

echo
echo "========================================"
echo "   YDS Setup - YOLO Dataset Studio"
echo "========================================"
echo

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is not installed or not in PATH"
  exit 1
fi

echo "[1/6] Python detected:"
python3 --version

VENV_DIR=".venv"

echo "[2/6] Creating virtual environment..."
if [ ! -x "${VENV_DIR}/bin/python" ]; then
  python3 -m venv "${VENV_DIR}"
else
  echo "Using existing venv: ${VENV_DIR}"
fi

echo "[3/6] Activating virtual environment..."
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "[4/6] Checking requirements.txt..."
if [ ! -f "requirements.txt" ]; then
  echo "ERROR: requirements.txt not found"
  exit 1
fi

echo "[5/6] Installing dependencies..."
python -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1
python -m pip install -r requirements.txt

echo "[6/6] Verifying installation..."
python -c "from ultralytics import YOLO; print('[OK] YOLO installed')" >/dev/null 2>&1

echo
echo "========================================"
echo "   Setup Complete!"
echo "========================================"
echo
echo "To launch the application, run:"
echo "   ./run.sh"
