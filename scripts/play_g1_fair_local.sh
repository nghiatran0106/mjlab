#!/usr/bin/env bash
# Play an audited fair-evaluation checkpoint on CPU with the Viser web viewer.
# Usage: bash scripts/play_g1_fair_local.sh [v7c|v7a|v7b|amp|expert|v4b|v7c1000]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

bundle="fair_results-20260928-040113/logs/rsl_rl/g1_velocity_amp"
case "${1:-v7c}" in
  v7c) checkpoint="$bundle/2026-09-28_03-44-34_gail-v7c/model_2000.pt" ;;
  v7a) checkpoint="$bundle/2026-09-28_03-44-34_gail-v7a/model_2000.pt" ;;
  v7b) checkpoint="$bundle/2026-09-28_03-44-34_gail-v7b/model_2000.pt" ;;
  amp) checkpoint="$bundle/2026-09-28_03-44-34_amp-v7d/model_2000.pt" ;;
  expert) checkpoint="checkpoints/demo/expert_step.pt" ;;
  v4b) checkpoint="checkpoints/demo/gail_v4b.pt" ;;
  v7c1000) checkpoint="checkpoints/demo/gail_v7c_it1000.pt" ;;
  *) echo "Choose: v7c, v7a, v7b, amp, expert, v4b, v7c1000" >&2; exit 2 ;;
esac

local_python="${MJLAB_PYTHON:-$HOME/venvs/mjlab/bin/python}"
[[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 1; }
[[ -x "$local_python" ]] || { echo "Missing Python: $local_python" >&2; exit 1; }
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MUJOCO_GL=disable CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/mjlab-uv-cache}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/mjlab-warp-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mjlab-matplotlib}"

echo "Playing ${1:-v7c}: $checkpoint"
# The Expert task has the matching actor inputs and command range, and loads
# only the actor through VelocityOnPolicyRunner. No discriminator/dataset/RSI
# file is required for inference. Play mode disables observation noise/pushes.
exec uv run --no-project --python "$local_python" python -m mjlab.scripts.play \
  Mjlab-Velocity-Flat-Unitree-G1-Expert \
  --checkpoint-file "$checkpoint" --device cpu --num-envs 1 --viewer viser
