#!/usr/bin/env bash
# Round 7: align the pipeline with the AMP/GAIL papers (see report plan).
#   All runs: stepping expert, reward = 0.3 * task + 0.7 * style (Escontrela et
#   al. 2022; before, style was only ~10% of the reward), 1M-transition replay
#   buffer for the discriminator, init_std 1, no entropy bonus.
#   A = key-body features                          (GAIL)
#   B = A + velocity command in the features       (conditional discriminator)
#   C = B + reference state initialization, 85%    (RSI)
#   D = C with the AMP least-squares loss instead of GAIL
# Works on a fresh machine (needs checkpoints/demo/expert_step.pt).
set -euo pipefail

expert="${EXPERT_CKPT:-checkpoints/demo/expert_step.pt}"
v4b="${V4B_CKPT:-checkpoints/demo/gail_v4b.pt}"
iters="${MAX_ITERATIONS:-5000}"
num_envs="${NUM_ENVS:-4096}"

export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MJLAB_INIT_STD=1.0 MJLAB_ENTROPY_COEF=0.0
export MJLAB_AMP_STYLE_WEIGHT=4 MJLAB_AMP_TASK_LERP=0.3 MJLAB_AMP_REPLAY=1000000
export MJLAB_AMP_DISC_UPDATES=5 MJLAB_AMP_DISC_LR=1e-4
uv_run=(uv run --frozen --no-dev --extra cu128)
mkdir -p logs/amp_console logs/amp_expert logs/amp_eval
stamp="$(date +%Y%m%d-%H%M%S)"

base=logs/amp_expert/v7_expert_base.npz
kb=logs/amp_expert/v7_expert_kb.npz
cond=logs/amp_expert/v7_expert_cond.npz
states=logs/amp_expert/v7_expert_states.npz

collect() {  # collect <task> <out> [extra args]
  [[ -f "$2" ]] && return
  "${uv_run[@]}" python -m mjlab.tasks.velocity_amp.scripts.collect_expert \
    "$expert" --task "$1" --out "$2" --num-envs 512 --steps 500 "${@:3}" \
    > "logs/amp_console/collect-$(basename "$2" .npz)-$stamp.log" 2>&1
}

train() {  # train <task> <run-name> [env assignments...]
  env "${@:3}" "${uv_run[@]}" train "$1" \
    --gpu-ids '[0]' \
    --env.scene.num-envs "$num_envs" \
    --agent.max-iterations "$iters" \
    --agent.save-interval 500 \
    --agent.logger tensorboard \
    --agent.upload-model False \
    --agent.run-name "$2" > "logs/amp_console/$2-$stamp.log" 2>&1
}

last() { echo "$(ls -td logs/rsl_rl/g1_velocity_amp/*_"$1" | head -1)/model_$((iters - 1)).pt"; }

echo "== expert data =="
collect Mjlab-Velocity-Flat-Unitree-G1-Expert "$base"
collect Mjlab-Velocity-Flat-Unitree-G1-Expert-KeyBody "$kb"
collect Mjlab-Velocity-Flat-Unitree-G1-Expert-Cond "$cond" --states-out "$states"
grep -h "Saved" logs/amp_console/collect-v7_*-"$stamp".log || true

echo "== training: 4 runs in parallel, $iters iterations =="
train Mjlab-Velocity-Flat-Unitree-G1-AMP-KeyBody gail-v7a \
  MJLAB_AMP_LOSS=gail MJLAB_AMP_EXPERT="$kb" &
train Mjlab-Velocity-Flat-Unitree-G1-AMP-Cond gail-v7b \
  MJLAB_AMP_LOSS=gail MJLAB_AMP_EXPERT="$cond" MJLAB_AMP_RSI_PROB=0 &
train Mjlab-Velocity-Flat-Unitree-G1-AMP-Cond gail-v7c \
  MJLAB_AMP_LOSS=gail MJLAB_AMP_EXPERT="$cond" \
  MJLAB_AMP_RSI_FILE="$states" MJLAB_AMP_RSI_PROB=0.85 &
train Mjlab-Velocity-Flat-Unitree-G1-AMP-Cond amp-v7d \
  MJLAB_AMP_LOSS=amp MJLAB_AMP_EXPERT="$cond" \
  MJLAB_AMP_RSI_FILE="$states" MJLAB_AMP_RSI_PROB=0.85 &
wait

echo "== evaluation =="
"${uv_run[@]}" python -m mjlab.tasks.velocity_amp.scripts.evaluate \
  --expert-file "$base" \
  --policy "expert_step=$expert" \
  --policy "gail_v4b=$v4b" \
  --policy "gail_v7a=$(last gail-v7a)" \
  --policy "gail_v7b=$(last gail-v7b)" \
  --policy "gail_v7c=$(last gail-v7c)" \
  --policy "amp_v7d=$(last amp-v7d)" \
  --out "logs/amp_eval/results-v7-$stamp.json" \
  > "logs/amp_console/eval-v7-$stamp.log" 2>&1
grep -E '^\| (policy|expert|gail|amp)' "logs/amp_console/eval-v7-$stamp.log"

echo "== snapshots =="
"${uv_run[@]}" python report/render_snapshots.py --out report/figures \
  --policy "gail_v7a=$(last gail-v7a)" --policy "gail_v7b=$(last gail-v7b)" \
  --policy "gail_v7c=$(last gail-v7c)" --policy "amp_v7d=$(last amp-v7d)" \
  > "logs/amp_console/snap-v7-$stamp.log" 2>&1 || echo "snapshots failed (see log)"

# Small bundle for a slow link: final checkpoints + results + images.
tar czf "logs/v7_results-$stamp.tgz" "logs/amp_eval/results-v7-$stamp.json" \
  logs/amp_console/*-v7*-"$stamp".log report/figures/snap_*v7*.png \
  "$(last gail-v7a)" "$(last gail-v7b)" "$(last gail-v7c)" "$(last amp-v7d)"
ls -la "logs/v7_results-$stamp.tgz"
echo "Done v7."
