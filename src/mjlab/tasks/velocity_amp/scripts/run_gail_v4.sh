#!/usr/bin/env bash
# Round 4: GAIL that walks with a natural gait.
#   GAIL v1 walked (speed ratio 0.94) but was jerky and crouched because the
#   entropy bonus pushed the action std to 3.5 and the discriminator diverged.
#   Keep GAIL + init_std 1.0 (enough exploration to find stepping), drop the
#   entropy bonus, and sweep discriminator strength x feature set.
# Works on a fresh machine: collects expert data if missing.
set -euo pipefail

expert_ckpt="${EXPERT_CKPT:-amp_inputs/expert_model_999.pt}"
num_envs="${NUM_ENVS:-4096}"
iters="${MAX_ITERATIONS:-1500}"
last_iter=$((iters - 1))

export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MJLAB_INIT_STD=1.0 MJLAB_ENTROPY_COEF=0.0 MJLAB_AMP_LOSS=gail
uv_run=(uv run --frozen --no-dev --extra cu128)
mkdir -p logs/amp_console logs/amp_expert
stamp="$(date +%Y%m%d-%H%M%S)"

collect() {  # collect <task> <out>
  [[ -f "$2" ]] && return
  "${uv_run[@]}" python -m mjlab.tasks.velocity_amp.scripts.collect_expert \
    "$expert_ckpt" --task "$1" --out "$2" --num-envs 512 --steps 500
}

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

latest() {  # latest <run-name> -> checkpoint path
  echo "$(ls -td logs/rsl_rl/g1_velocity_amp/*_"$1" | head -1)/model_${last_iter}.pt"
}

echo "== expert data =="
base=logs/amp_expert/g1_velocity_expert.npz
kb=logs/amp_expert/g1_velocity_expert_kb.npz
collect Mjlab-Velocity-Flat-Unitree-G1-Expert "$base"
collect Mjlab-Velocity-Flat-Unitree-G1-Expert-KeyBody "$kb"

echo "== GAIL v4 training (4 runs in parallel, $iters iterations) =="
# a/b: moderate discriminator (5 steps, lr 1e-4), style weight 2.
# c/d: weak discriminator (2 steps, lr 3e-5), style weight 1.
MJLAB_AMP_EXPERT=$base MJLAB_AMP_STYLE_WEIGHT=2 MJLAB_AMP_DISC_UPDATES=5 \
  MJLAB_AMP_DISC_LR=1e-4 train Mjlab-Velocity-Flat-Unitree-G1-AMP gail-v4a &
MJLAB_AMP_EXPERT=$kb MJLAB_AMP_STYLE_WEIGHT=2 MJLAB_AMP_DISC_UPDATES=5 \
  MJLAB_AMP_DISC_LR=1e-4 train Mjlab-Velocity-Flat-Unitree-G1-AMP-KeyBody gail-v4b &
MJLAB_AMP_EXPERT=$base MJLAB_AMP_STYLE_WEIGHT=1 MJLAB_AMP_DISC_UPDATES=2 \
  MJLAB_AMP_DISC_LR=3e-5 train Mjlab-Velocity-Flat-Unitree-G1-AMP gail-v4c &
MJLAB_AMP_EXPERT=$kb MJLAB_AMP_STYLE_WEIGHT=1 MJLAB_AMP_DISC_UPDATES=2 \
  MJLAB_AMP_DISC_LR=3e-5 train Mjlab-Velocity-Flat-Unitree-G1-AMP-KeyBody gail-v4d &
wait

echo "== evaluation =="
"${uv_run[@]}" python -m mjlab.tasks.velocity_amp.scripts.evaluate \
  --expert-file "$base" \
  --policy "expert=$expert_ckpt" \
  --policy "gail_v4a=$(latest gail-v4a)" \
  --policy "gail_v4b=$(latest gail-v4b)" \
  --policy "gail_v4c=$(latest gail-v4c)" \
  --policy "gail_v4d=$(latest gail-v4d)" \
  --out "logs/amp_eval/results-v4-$stamp.json" \
  > "logs/amp_console/eval-v4-$stamp.log" 2>&1
grep -E '^\| ' "logs/amp_console/eval-v4-$stamp.log"

echo "== snapshots =="
"${uv_run[@]}" python report/render_snapshots.py --out report/figures \
  --policy "gail_v4a=$(latest gail-v4a)" --policy "gail_v4b=$(latest gail-v4b)" \
  --policy "gail_v4c=$(latest gail-v4c)" --policy "gail_v4d=$(latest gail-v4d)" \
  > "logs/amp_console/snap-v4-$stamp.log" 2>&1 || echo "snapshots failed (see log)"
echo "Done v4."
