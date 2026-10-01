#!/usr/bin/env bash
# Play a downloaded G1 policy on CPU using the existing local environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

checkpoint="${1:-logs/rsl_rl/g1_velocity/2026-09-20_15-35-48_vast-train/model_999.pt}"
if [[ ! -f "$checkpoint" ]]; then
  echo "Missing checkpoint: $checkpoint" >&2
  echo "Download and extract mjlab-g1-results.tar.gz from Vast first." >&2
  exit 1
fi

local_python="${MJLAB_PYTHON:-$HOME/venvs/mjlab/bin/python}"
if [[ ! -x "$local_python" ]]; then
  echo "Set MJLAB_PYTHON to the Python executable of your mjlab environment." >&2
  exit 1
fi

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL=disable
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/mjlab-uv-cache}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/mjlab-warp-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mjlab-matplotlib}"

# TASK=Mjlab-Velocity-Flat-Unitree-G1-Expert plays with the training command
# ranges (the default task's play mode samples faster commands).
exec uv run --no-project --python "$local_python" python -m mjlab.scripts.play \
  "${TASK:-Mjlab-Velocity-Flat-Unitree-G1}" \
  --checkpoint-file "$checkpoint" --device cpu --num-envs 1 --viewer viser
