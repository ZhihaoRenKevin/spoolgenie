#!/usr/bin/env bash
# Zero-interaction macOS & Linux installer for SpoolGenie (ideal for humans & AI Agents).
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/ZhihaoRenKevin/spoolgenie/main/scripts/install.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/ZhihaoRenKevin/spoolgenie/main/scripts/install.sh | bash -s -- --gpu --launch

set -euo pipefail

REPO_URL="https://github.com/ZhihaoRenKevin/spoolgenie.git"
INSTALL_DIR="${SPOOLGENIE_INSTALL_DIR:-$HOME/spoolgenie}"
WANT_GPU=0
WANT_DESKTOP=1
WANT_LAUNCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir) INSTALL_DIR="$2"; shift 2 ;;
    --gpu) WANT_GPU=1; shift ;;
    --desktop) WANT_DESKTOP=1; shift ;;
    --no-desktop) WANT_DESKTOP=0; shift ;;
    --launch) WANT_LAUNCH=1; shift ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

echo ""
echo "[SpoolGenie] Starting automated macOS/Linux local deployment..."

if [[ -f "src/spoolgenie/__init__.py" ]]; then
  TARGET_DIR="$(pwd)"
  echo "[SpoolGenie] Detected existing checkout: $TARGET_DIR"
else
  TARGET_DIR="$INSTALL_DIR"
  if [[ -d "$TARGET_DIR/.git" ]]; then
    echo "[SpoolGenie] Updating existing repo in $TARGET_DIR ..."
    git -C "$TARGET_DIR" pull --ff-only
  else
    echo "[SpoolGenie] Cloning $REPO_URL -> $TARGET_DIR ..."
    git clone "$REPO_URL" "$TARGET_DIR"
  fi
fi

cd "$TARGET_DIR"

EXTRAS="dev"
if [[ "$WANT_DESKTOP" -eq 1 ]]; then
  EXTRAS="dev,desktop"
fi

if command -v uv >/dev/null 2>&1; then
  echo "[SpoolGenie] Using uv to install SpoolGenie[$EXTRAS]..."
  [[ -d ".venv" ]] || uv venv
  uv pip install -e ".[$EXTRAS]"
else
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
  if [[ -z "$PYTHON_BIN" ]]; then
    echo "[SpoolGenie] Error: Python 3.10-3.12 is required." >&2
    exit 1
  fi
  echo "[SpoolGenie] Using $PYTHON_BIN -m venv to install SpoolGenie[$EXTRAS]..."
  [[ -d ".venv" ]] || "$PYTHON_BIN" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/pip install -e ".[$EXTRAS]"
fi

if [[ "$WANT_GPU" -eq 1 ]]; then
  echo "[SpoolGenie] Unlocking local GPU (Apple MPS / CUDA) + SDXL + Hunyuan3D-2..."
  .venv/bin/python scripts/setup_env.py --gpu --local-2d --local-3d
fi

echo ""
echo "[SpoolGenie] Verifying hardware & environment..."
.venv/bin/spoolgenie doctor

echo ""
echo "[SpoolGenie] AI Agent MCP integration snippet:"
.venv/bin/spoolgenie agent-config

echo ""
echo "[SpoolGenie] Deployment complete!"
echo "  Desktop App : $TARGET_DIR/.venv/bin/spoolgenie desktop"
echo "  Web + Phone : $TARGET_DIR/.venv/bin/spoolgenie serve --lan --port 8765"
echo "  MCP Server  : $TARGET_DIR/.venv/bin/spoolgenie-mcp"

if [[ "$WANT_LAUNCH" -eq 1 ]]; then
  echo ""
  echo "[SpoolGenie] Launching Web/PWA Studio with LAN QR pairing on port 8765..."
  exec .venv/bin/spoolgenie serve --lan --port 8765
fi
