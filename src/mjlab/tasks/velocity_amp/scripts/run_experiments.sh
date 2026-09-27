#!/usr/bin/env bash
# Full AMP experiment on one NVIDIA GPU instance (about 45-60 min on a 4090).
#
#   1. collect expert transitions from the hand-shaped PPO policy,
#   2. train the task-only baseline (velocity tracking reward only),
#   3. train AMP (velocity tracking + learned style reward),
#   4. evaluate expert / task-only / AMP under the original reward.
#
# Usage (from the repo root, ideally inside tmux):
#   EXPERT_CKPT=amp_inputs/expert_model_999.pt \
#     bash src/mjlab/tasks/velocity_amp/scripts/run_experiments.sh
# Overrides: NUM_ENVS (4096), MAX_ITERATIONS (1000), WITH_GAIL=1 adds a GAIL run.
set -euo pipefail

expert_ckpt="${EXPERT_CKPT:?Set EXPERT_CKPT to the expert model_*.pt}"
num_envs="${NUM_ENVS:-4096}"
iters="${MAX_ITERATIONS:-1000}"
last_iter=$((iters - 1))
expert_file="logs/amp_expert/g1_velocity_expert.npz"

export MUJOCO_GL=egl PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
uv_run=(uv run --frozen --no-dev --extra cu128)
mkdir -p logs/amp_console
stamp="$(date +%Y%m%d-%H%M%S)"

nvidia-smi --query-gpu=name,memory.total --format=csv
"${uv_run[@]}" python -c 'import torch; assert torch.cuda.is_available(); print("CUDA ok")'

train() {  # train <task> <run-name>
  "${uv_run[@]}" train "$1" \
    --gpu-ids '[0]' \
    --env.scene.num-envs "$num_envs" \
    --agent.max-iterations "$iters" \
    --agent.save-interval 100 \
    --agent.logger tensorboard \
    --agent.upload-model False \
    --agent.run-name "$2" 2>&1 | tee "logs/amp_console/$2-$stamp.log"
}

latest() {  # latest <experiment_name> <run-name> -> checkpoint path
  local dir
  dir="$(ls -td logs/rsl_rl/"$1"/*_"$2" | head -1)"
  echo "$dir/model_${last_iter}.pt"
}

echo "== 1/4 expert data =="
"${uv_run[@]}" python -m mjlab.tasks.velocity_amp.scripts.collect_expert \
  "$expert_ckpt" --out "$expert_file" --num-envs 512 --steps 500 \
  2>&1 | tee "logs/amp_console/expert-$stamp.log"

echo "== 2/4 task-only baseline =="
train Mjlab-Velocity-Flat-Unitree-G1-TaskOnly vast-task-only

echo "== 3/4 AMP =="
export MJLAB_AMP_EXPERT="$expert_file" MJLAB_AMP_LOSS=amp
train Mjlab-Velocity-Flat-Unitree-G1-AMP vast-amp

policies=(
  --policy "expert=$expert_ckpt"
  --policy "task_only=$(latest g1_velocity_task_only vast-task-only)"
  --policy "amp=$(latest g1_velocity_amp vast-amp)"
)
if [[ "${WITH_GAIL:-0}" == "1" ]]; then
  echo "== 3b GAIL =="
  export MJLAB_AMP_LOSS=gail
  train Mjlab-Velocity-Flat-Unitree-G1-AMP vast-gail
  policies+=(--policy "gail=$(latest g1_velocity_amp vast-gail)")
fi

echo "== 4/4 evaluation =="
"${uv_run[@]}" python -m mjlab.tasks.velocity_amp.scripts.evaluate \
  --expert-file "$expert_file" "${policies[@]}" \
  --out "logs/amp_eval/results-$stamp.json" \
  2>&1 | tee "logs/amp_console/eval-$stamp.log"
grep -E '^\| (policy|---|[a-z_]+ \|)' "logs/amp_console/eval-$stamp.log"

echo "Done. Copy logs/ back before destroying the instance."
