#!/usr/bin/env bash
# Run inside an existing NVIDIA GPU instance; see VAST_TRAINING_VI.md.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

case "${1:-help}" in
  smoke) default_envs=128; default_iterations=10 ;;
  benchmark) default_envs=4096; default_iterations=50 ;;
  train) default_envs=4096; default_iterations=1000 ;;
  *)
    echo "Usage: bash scripts/cloud/train_vast.sh {smoke|benchmark|train} [train flags]"
    echo "Overrides: NUM_ENVS, MAX_ITERATIONS. Logs use local TensorBoard."
    exit 0
    ;;
esac
stage="$1"
shift

command -v uv >/dev/null || { echo "Install uv first." >&2; exit 1; }
command -v nvidia-smi >/dev/null || {
  echo "An NVIDIA GPU instance is required. See VAST_TRAINING_VI.md." >&2
  exit 1
}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
uv_run=(uv run --frozen --no-dev --extra cu128)
"${uv_run[@]}" python -c '
import torch
import warp as wp
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA; check instance driver and GPU access.")
wp.init()
if not wp.is_cuda_available():
    raise SystemExit("Warp cannot access CUDA; check the host NVIDIA driver.")
x = torch.ones(1, device="cuda:0")
torch.cuda.synchronize()
print("CUDA ready:", torch.cuda.get_device_name(0), "PyTorch", torch.__version__)
'

mkdir -p logs/vast
log_file="logs/vast/${stage}-$(date +%Y%m%d-%H%M%S)-$$.log"
"${uv_run[@]}" train Mjlab-Velocity-Flat-Unitree-G1 \
  --gpu-ids '[0]' \
  --env.scene.num-envs "${NUM_ENVS:-$default_envs}" \
  --agent.max-iterations "${MAX_ITERATIONS:-$default_iterations}" \
  --agent.save-interval 50 \
  --agent.logger tensorboard \
  --agent.upload-model False \
  --agent.run-name "vast-${stage}" \
  "$@" 2>&1 | tee "$log_file"

echo "Completed ${stage}. Console log: ${log_file}"
echo "Checkpoints: logs/rsl_rl/g1_velocity/. Copy logs off the instance before deletion."
