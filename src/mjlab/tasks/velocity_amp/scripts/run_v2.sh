#!/usr/bin/env bash
# Round 2: bounded exploration + weaker discriminator (see report, section 7).
# Trains task-only-v2 and AMP-v2 in parallel on one GPU, then evaluates all
# policies from both rounds. Run from the repo root after run_experiments.sh.
set -euo pipefail

expert_ckpt="${EXPERT_CKPT:?Set EXPERT_CKPT to the expert model_*.pt}"
expert_file="logs/amp_expert/g1_velocity_expert.npz"
num_envs="${NUM_ENVS:-4096}"
iters="${MAX_ITERATIONS:-1000}"
last_iter=$((iters - 1))

export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MJLAB_INIT_STD=0.3 MJLAB_ENTROPY_COEF=0.0
uv_run=(uv run --frozen --no-dev --extra cu128)
mkdir -p logs/amp_console
stamp="$(date +%Y%m%d-%H%M%S)"

train() {  # train <task> <run-name>
  "${uv_run[@]}" train "$1" \
    --gpu-ids '[0]' \
    --env.scene.num-envs "$num_envs" \
    --agent.max-iterations "$iters" \
    --agent.save-interval 100 \
    --agent.logger tensorboard \
    --agent.upload-model False \
    --agent.run-name "$2" > "logs/amp_console/$2-$stamp.log" 2>&1
}

latest() {  # latest <experiment_name> <run-name> -> checkpoint path
  local dir
  dir="$(ls -td logs/rsl_rl/"$1"/*_"$2" | head -1)"
  echo "$dir/model_${last_iter}.pt"
}

echo "== v2 training (parallel) =="
train Mjlab-Velocity-Flat-Unitree-G1-TaskOnly vast-task-only-v2 &
pid_task=$!
MJLAB_AMP_EXPERT="$expert_file" MJLAB_AMP_LOSS=amp MJLAB_AMP_STYLE_WEIGHT=4.0 \
  MJLAB_AMP_DISC_LR=1e-5 MJLAB_AMP_DISC_UPDATES=2 \
  train Mjlab-Velocity-Flat-Unitree-G1-AMP vast-amp-v2 &
pid_amp=$!
wait "$pid_task"
wait "$pid_amp"

echo "== evaluation =="
"${uv_run[@]}" python -m mjlab.tasks.velocity_amp.scripts.evaluate \
  --expert-file "$expert_file" \
  --policy "expert=$expert_ckpt" \
  --policy "task_only=$(latest g1_velocity_task_only vast-task-only)" \
  --policy "amp=$(latest g1_velocity_amp vast-amp)" \
  --policy "gail=$(latest g1_velocity_amp vast-gail)" \
  --policy "task_only_v2=$(latest g1_velocity_task_only vast-task-only-v2)" \
  --policy "amp_v2=$(latest g1_velocity_amp vast-amp-v2)" \
  --out "logs/amp_eval/results-v2-$stamp.json" \
  > "logs/amp_console/eval-v2-$stamp.log" 2>&1
grep -E '^\| ' "logs/amp_console/eval-v2-$stamp.log"
echo "Done v2."
