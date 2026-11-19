# Source from the project root before working:  source scripts/env.sh
# Sets privacy defaults and activates the project venv. Holds no secrets.

export TELEMETRY_OPT_OUT=true        # Fireworks training-skill analytics off
export HF_HUB_DISABLE_TELEMETRY=1    # Hugging Face Hub telemetry off
export DO_NOT_TRACK=1
export WANDB_MODE=disabled           # no Weights & Biases logging

_PA_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
if [ -f "$_PA_ROOT/vendor/cookbook/training/.venv/bin/activate" ]; then
  . "$_PA_ROOT/vendor/cookbook/training/.venv/bin/activate"
else
  echo "venv missing: run scripts/setup_cookbook.sh" >&2
fi
[ -n "${FIREWORKS_API_KEY:-}" ] || echo "FIREWORKS_API_KEY is not set in this shell" >&2
unset _PA_ROOT
