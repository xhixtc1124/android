#!/usr/bin/env bash
# Recreate the Fireworks cookbook at a pinned commit and build its venv.
# Usage: scripts/setup_cookbook.sh   (run from anywhere; safe to re-run)
set -euo pipefail

COOKBOOK_REPO="https://github.com/fw-ai-external/cookbook.git"
COOKBOOK_COMMIT="74e103ab4249d508912ccf264793e655980eead0"  # 2026-11-12, cookbook 2026.11.13.1
PYTHON_VERSION="3.11"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COOKBOOK="$ROOT/vendor/cookbook"
VENV="$COOKBOOK/training/.venv"

if [ ! -d "$COOKBOOK/.git" ]; then
  mkdir -p "$ROOT/vendor"
  git clone --quiet "$COOKBOOK_REPO" "$COOKBOOK"
fi
git -C "$COOKBOOK" fetch --quiet origin "$COOKBOOK_COMMIT" 2>/dev/null || true
git -C "$COOKBOOK" -c advice.detachedHead=false checkout --quiet "$COOKBOOK_COMMIT"
echo "cookbook at $(git -C "$COOKBOOK" rev-parse --short HEAD)"

if [ ! -x "$VENV/bin/python" ]; then
  uv venv --quiet --python "$PYTHON_VERSION" "$VENV"
fi
# Cookbook package (editable), test runner, and this project's package (editable).
uv pip install --quiet --python "$VENV/bin/python" -e "$COOKBOOK/training" pytest
if [ -f "$ROOT/pyproject.toml" ]; then
  uv pip install --quiet --python "$VENV/bin/python" -e "$ROOT"
fi
echo "venv ready: $VENV ($("$VENV/bin/python" --version))"
