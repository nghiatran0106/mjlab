#!/usr/bin/env bash
# Phase 2 of the optimization plan (after run_phase1.sh, same seeds).
# Phase 1 showed a trade-off: a task-heavy reward (RSI only) learns to walk in
# 250-500 iterations but shuffles; a style-heavy reward (v7c) gives the expert
# gait but one seed in three collapsed to standing still. Both runs here start
# from v7c and schedule the task-reward weight lambda: 0.9 for 250 iterations,
# then linearly to 0.3 by iteration 750.
#   - sched:    v7c + lambda schedule;
#   - sched_bc: sched + behavior-cloning pretraining of the actor on the expert.
# Compare iterations-to-target with v7c from phase 1 (same seeds, same protocol).
# Works on a fresh machine (needs checkpoints/demo/expert_step.pt).
set -euo pipefail

expert="${EXPERT_CKPT:-checkpoints/demo/expert_step.pt}"
iters="${MAX_ITERATIONS:-1500}"
every="${EVAL_EVERY:-250}"
num_envs="${NUM_ENVS:-4096}"
seeds=(${SEEDS:-1 2 3})
# GPUs for the two configs; with two GPUs (e.g. Kaggle 2x T4) each config gets
# its own device, with one GPU both share it.
gpus=(${GPUS:-0 0})

export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MJLAB_INIT_STD=1.0 MJLAB_ENTROPY_COEF=0.0 MJLAB_AMP_LOSS=gail
export MJLAB_AMP_DISC_UPDATES=5 MJLAB_AMP_DISC_LR=1e-4
uv_run=(uv run --frozen --no-dev --extra cu128)
mkdir -p logs/amp_console logs/amp_expert logs/amp_eval
stamp="$(date +%Y%m%d-%H%M%S)"

base=logs/amp_expert/v7_expert_base.npz
cond=logs/amp_expert/p2_expert_cond.npz
states=logs/amp_expert/p2_expert_states.npz
bc=logs/amp_expert/p2_expert_bc.npz

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
collect Mjlab-Velocity-Flat-Unitree-G1-Expert-Cond "$cond" --states-out "$states" \
  --bc-out "$bc"

echo "== training: {sched, sched_bc} x ${#seeds[@]} seeds, $iters iterations =="
common=(MJLAB_AMP_EXPERT="$cond" MJLAB_AMP_STYLE_WEIGHT=4 MJLAB_AMP_TASK_LERP=0.3
  MJLAB_AMP_LERP_START=0.9 MJLAB_AMP_LERP_HOLD=250 MJLAB_AMP_LERP_RAMP=500
  MJLAB_AMP_REPLAY=1000000 MJLAB_AMP_RSI_FILE="$states" MJLAB_AMP_RSI_PROB=0.85)
runs=()
for s in "${seeds[@]}"; do
  train Mjlab-Velocity-Flat-Unitree-G1-AMP-Cond "p2-sched-s$s" "$s" "${common[@]}" \
    CUDA_VISIBLE_DEVICES="${gpus[0]}" &
  train Mjlab-Velocity-Flat-Unitree-G1-AMP-Cond "p2-schedbc-s$s" "$s" "${common[@]}" \
    MJLAB_AMP_BC_FILE="$bc" MJLAB_AMP_BC_STEPS=2000 CUDA_VISIBLE_DEVICES="${gpus[1]}" &
  runs+=("p2-sched-s$s" "p2-schedbc-s$s")
done
wait

echo "== evaluation of every checkpoint =="
policies=(--policy "expert_step=$expert")
for r in "${runs[@]}"; do
  name="${r#p2-}"            # sched-s1 / schedbc-s1
  config="${name%-s*}"       # sched / schedbc
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
  --out "logs/amp_eval/results-phase2-$stamp.json" \
  > "logs/amp_console/eval-phase2-$stamp.log" 2>&1
"${uv_run[@]}" python -m mjlab.tasks.velocity_amp.scripts.convergence \
  "logs/amp_eval/results-phase2-$stamp.json" \
  --out "logs/amp_eval/convergence-phase2-$stamp.json" | tee "logs/phase2_summary.md"

# Small bundle for a slow link: results, summary, logs, TensorBoard events and
# the final checkpoint of each run.
files=("logs/amp_eval/results-phase2-$stamp.json" "logs/amp_eval/convergence-phase2-$stamp.json"
  logs/phase2_summary.md logs/amp_console/*-"$stamp".log)
for r in "${runs[@]}"; do
  d="$(run_dir "$r")"
  files+=("$d"/events.out.* "$(ls -t "$d"/model_*.pt | head -1)")
done
tar czf "logs/phase2_results-$stamp.tgz" "${files[@]}"
ls -la "logs/phase2_results-$stamp.tgz"
echo "Done phase 2."
