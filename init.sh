#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/voloha-mil/Bollards.git}"
REPO_DIR="${REPO_DIR:-Bollards}"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQ_FILE="${REQ_FILE:-requirements.txt}"

echo "[info] Updating apt and installing system deps..."
sudo apt-get update
sudo apt-get install -y \
  git \
  ${PYTHON_BIN}-venv \
  python3-pip \
  build-essential \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender1

echo "[info] Cloning/updating repo..."
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

echo "[info] Creating venv..."
$PYTHON_BIN -m venv "$VENV_DIR"

echo "[info] Activating venv..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[info] Upgrading pip tooling..."
pip install -U pip setuptools wheel

echo "[info] Installing Python requirements..."
if [ -f "$REQ_FILE" ]; then
  pip install -r "$REQ_FILE"
else
  echo "[warn] $REQ_FILE not found, skipping."
fi

echo "[info] Ensuring runtime deps..."
pip install -U boto3

echo "[info] Quick import checks..."
python - <<'PY'
import sys
mods = ["torch", "ultralytics", "huggingface_hub", "pandas", "boto3"]
bad = []
for m in mods:
    try:
        __import__(m)
    except Exception as e:
        bad.append((m, str(e)))
if bad:
    print("[error] Import failures:")
    for m, e in bad:
        print(f"  - {m}: {e}")
    sys.exit(1)
print("[ok] All imports succeeded.")
PY

echo
echo "[done] Instance prepared."
echo "Next:"
echo "  cd $REPO_DIR && source $VENV_DIR/bin/activate"
echo "  python mine_osv5m_bollards.py --help"
