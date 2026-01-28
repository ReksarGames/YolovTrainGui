#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "ERROR: .venv not found. Run ./setup.sh first."
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
python GUI.py
