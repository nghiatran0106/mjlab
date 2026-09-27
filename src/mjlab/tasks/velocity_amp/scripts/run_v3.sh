#!/usr/bin/env bash
# Round 3: aim for the closest match to the expert.
#   - exploration like the expert early (init_std 1.0) but no entropy bonus,
#     so the action std can shrink instead of being pushed up (v1 failure),
#   - weak discriminator (v2 settings), style weight 4,
#   - two feature sets in parallel: base (68-d) and + feet/hands (80-d),
#   - 3000 iterations.
set -euo pipefail

expert_ckpt="${EXPERT_CKPT:?Set EXPERT_CKPT to the expert model_*.pt}"
num_envs="${NUM_ENVS:-4096}"
iters="${MAX_ITERATIONS:-3000}"
last_iter=$((iters - 1))

export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MJLAB_INIT_STD=1.0 MJLAB_ENTROPY_COEF=0.0
export MJLAB_AMP_LOSS=amp MJLAB_AMP_STYLE_WEIGHT=4.0
export MJLAB_AMP_DISC_LR=1e-5 MJLAB_AMP_DISC_UPDATES=2
uv_run=(uv run --frozen --no-dev --extra cu128)
mkdir -p logs/amp_console
stamp="$(date +%Y%m%d-%H%M%S)"

train() {  # train <task> <run-name>
  "${uv_run[@]}" train "$1" \
    --gpu-ids '[0]' \
    --env.scene.num-envs "$num_envs" \
    --agent.max-iterations "$iters" \
    --agent.save-interval 250 \
    --agent.logger tensorboard \
    --agent.upload-model False \
    --agent.run-name "$2" > "logs/amp_console/$2-$stamp.log" 2>&1
}

latest() {  # latest <experiment_name> <run-name> [iter] -> checkpoint path
  local dir
  dir="$(ls -td logs/rsl_rl/"$1"/*_"$2" | head -1)"
  echo "$dir/model_${3:-$last_iter}.pt"
}

echo "== key-body expert data =="
"${uv_run[@]}" python -m mjlab.tasks.velocity_amp.scripts.collect_expert \
  "$expert_ckpt" --task Mjlab-Velocity-Flat-Unitree-G1-Expert-KeyBody \
  --out logs/amp_expert/g1_velocity_expert_kb.npz --num-envs 512 --steps 500 \
  > "logs/amp_console/expert-kb-$stamp.log" 2>&1
tail -2 "logs/amp_console/expert-kb-$stamp.log"

echo "== v3 training (parallel, $iters iterations) =="
MJLAB_AMP_EXPERT=logs/amp_expert/g1_velocity_expert.npz \
  train Mjlab-Velocity-Flat-Unitree-G1-AMP vast-amp-v3 &
pid_a=$!
MJLAB_AMP_EXPERT=logs/amp_expert/g1_velocity_expert_kb.npz \
  train Mjlab-Velocity-Flat-Unitree-G1-AMP-KeyBody vast-amp-v3-kb &
pid_b=$!
wait "$pid_a"
wait "$pid_b"

echo "== evaluation =="
"${uv_run[@]}" python -m mjlab.tasks.velocity_amp.scripts.evaluate \
  --expert-file logs/amp_expert/g1_velocity_expert.npz \
  --policy "expert=$expert_ckpt" \
  --policy "task_only=$(latest g1_velocity_task_only vast-task-only 999)" \
  --policy "amp=$(latest g1_velocity_amp vast-amp 999)" \
  --policy "gail=$(latest g1_velocity_amp vast-gail 999)" \
  --policy "task_only_v2=$(latest g1_velocity_task_only vast-task-only-v2 999)" \
  --policy "amp_v2=$(latest g1_velocity_amp vast-amp-v2 999)" \
  --policy "amp_v3=$(latest g1_velocity_amp vast-amp-v3)" \
  --policy "amp_v3_kb=$(latest g1_velocity_amp vast-amp-v3-kb)" \
  --out "logs/amp_eval/results-v3-$stamp.json" \
  > "logs/amp_console/eval-v3-$stamp.log" 2>&1
grep -E '^\| ' "logs/amp_console/eval-v3-$stamp.log"
echo "Done v3."
