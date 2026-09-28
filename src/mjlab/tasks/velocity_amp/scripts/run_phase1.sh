#!/usr/bin/env bash
# Phase 1 of the optimization plan: a reliable baseline.
#   - v7c (full round-7 config) with 3 seeds;
#   - "rsi_only": GAIL v4b settings (additive reward, no replay buffer, key-body
#     features) + RSI, to test whether RSI alone is enough.
# Checkpoints every 250 iterations are all evaluated with the common protocol,
# then convergence.py reports the iterations needed to reach the target.
# Works on a fresh machine (needs checkpoints/demo/expert_step.pt).
set -euo pipefail

expert="${EXPERT_CKPT:-checkpoints/demo/expert_step.pt}"
iters="${MAX_ITERATIONS:-2000}"
every="${EVAL_EVERY:-250}"
num_envs="${NUM_ENVS:-4096}"
seeds=(${SEEDS:-1 2 3})

export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MJLAB_INIT_STD=1.0 MJLAB_ENTROPY_COEF=0.0 MJLAB_AMP_LOSS=gail
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

train() {  # train <task> <run-name> <seed> [env assignments...]
  env "${@:4}" "${uv_run[@]}" train "$1" \
    --gpu-ids '[0]' \
    --env.scene.num-envs "$num_envs" \
    --agent.max-iterations "$iters" \
    --agent.save-interval "$every" \
    --agent.seed "$3" \
    --agent.logger tensorboard \
    --agent.upload-model False \
    --agent.run-name "$2" > "logs/amp_console/$2-$stamp.log" 2>&1
}

run_dir() { ls -td logs/rsl_rl/g1_velocity_amp/*_"$1" | head -1; }

echo "== expert data =="
collect Mjlab-Velocity-Flat-Unitree-G1-Expert "$base"
collect Mjlab-Velocity-Flat-Unitree-G1-Expert-KeyBody "$kb"
collect Mjlab-Velocity-Flat-Unitree-G1-Expert-Cond "$cond" --states-out "$states"

echo "== training: v7c x ${#seeds[@]} seeds + rsi_only, $iters iterations =="
runs=()
for s in "${seeds[@]}"; do
  train Mjlab-Velocity-Flat-Unitree-G1-AMP-Cond "p1-v7c-s$s" "$s" \
    MJLAB_AMP_EXPERT="$cond" MJLAB_AMP_STYLE_WEIGHT=4 MJLAB_AMP_TASK_LERP=0.3 \
    MJLAB_AMP_REPLAY=1000000 MJLAB_AMP_RSI_FILE="$states" MJLAB_AMP_RSI_PROB=0.85 &
  runs+=("p1-v7c-s$s")
done
train Mjlab-Velocity-Flat-Unitree-G1-AMP-KeyBody "p1-rsionly-s1" 1 \
  MJLAB_AMP_EXPERT="$kb" MJLAB_AMP_STYLE_WEIGHT=2 MJLAB_AMP_TASK_LERP=-1 \
  MJLAB_AMP_REPLAY=0 MJLAB_AMP_RSI_FILE="$states" MJLAB_AMP_RSI_PROB=0.85 &
runs+=("p1-rsionly-s1")
wait

echo "== evaluation of every checkpoint =="
policies=(--policy "expert_step=$expert")
for r in "${runs[@]}"; do
  name="${r#p1-}"            # v7c-s1 / rsionly-s1
  config="${name%-s*}"       # v7c / rsionly
  seed="${name##*-s}"
  d="$(run_dir "$r")"
  for ((it = every; it <= iters; it += every)); do
    ckpt="$d/model_$it.pt"
    [[ -f "$ckpt" ]] || ckpt="$d/model_$((it - 1)).pt"
    [[ -f "$ckpt" ]] && policies+=(--policy "${config}_s${seed}_it${it}=$ckpt")
  done
done
"${uv_run[@]}" python -m mjlab.tasks.velocity_amp.scripts.evaluate \
  --expert-file "$base" "${policies[@]}" \
  --out "logs/amp_eval/results-phase1-$stamp.json" \
  > "logs/amp_console/eval-phase1-$stamp.log" 2>&1
"${uv_run[@]}" python -m mjlab.tasks.velocity_amp.scripts.convergence \
  "logs/amp_eval/results-phase1-$stamp.json" \
  --out "logs/amp_eval/convergence-phase1-$stamp.json" | tee "logs/phase1_summary.md"

# Small bundle for a slow link: results, summary, logs, TensorBoard events and
# the final checkpoint of each run.
files=("logs/amp_eval/results-phase1-$stamp.json" "logs/amp_eval/convergence-phase1-$stamp.json"
  logs/phase1_summary.md logs/amp_console/*-"$stamp".log)
for r in "${runs[@]}"; do
  d="$(run_dir "$r")"
  files+=("$d"/events.out.* "$(ls -t "$d"/model_*.pt | head -1)")
done
tar czf "logs/phase1_results-$stamp.tgz" "${files[@]}"
ls -la "logs/phase1_results-$stamp.tgz"
echo "Done phase 1."
