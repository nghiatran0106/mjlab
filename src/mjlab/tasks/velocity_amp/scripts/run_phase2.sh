#!/usr/bin/env bash
# Phase 2 of the optimization plan (after run_phase1.sh, same seeds).
# Phase 1 showed a trade-off: a task-heavy reward (RSI only) learns to walk in
# 250-500 iterations but shuffles; a style-heavy reward (v7c) gives the expert
# gait but one seed in three collapsed to standing still. Both runs here start
# from v7c and schedule the task-reward weight lambda: 0.9 for 250 iterations,
# then linearly to 0.3 by iteration 750.
#   - sched:    v7c + lambda schedule;
#   - schedbc:  sched + behavior-cloning pretraining of the actor on the expert;
#   - bc:       v7c (constant lambda = 0.3) + behavior cloning. Added after seed 1
#               of sched/schedbc: the task-heavy start made actions jerky, while
#               behavior cloning alone already gave an expert-like gait at start.
#   - ppo:      baseline, standard PPO with the 14 hand-written reward terms trained
#               from scratch (the ExpertStep config that produced the expert);
#   - bctask:   bc without the GAIL style reward (task reward only), to separate
#               the contribution of GAIL from that of behavior cloning.
#   - bcr:      bc + action-rate penalty as a third hand-written term. The
#               discriminator sees states only, and bc loses 6 of its 9 points to
#               the expert on action jerk. The weight is -0.1 after the 0.3 task
#               weight, the same as in the hand-written reward;
#   - bcrs:     bcr + initial action std 0.3 instead of 1.0, so exploration noise
#               does not undo the cloned policy in the first ~150 iterations;
#   - bcrsw:    bcrs + 25 iterations that update only the critic (it starts random
#               after behavior cloning).
# CONFIGS selects the configs (default "sched schedbc"). Compare
# iterations-to-target with v7c from phase 1 (same seeds, same protocol).
# With TIMING=1 (default) every config in TIMING_CONFIGS (default: CONFIGS) is
# also timed alone on one GPU for TIMING_ITERS iterations, so seconds per
# iteration are comparable across configs.
# VARIANT=payload|weak|slippery runs the reuse experiment instead: the configs
# are trained on that robot variant (6 kg torso payload, PD gains at 70%, low
# foot friction) while the demonstrations still come from the nominal expert.
# Iterations-to-target are then measured against the converged "ppo" runs on
# the same variant (convergence.py --reference-config ppo), so CONFIGS must
# include ppo, e.g. VARIANT=payload CONFIGS="ppo bc".
# Works on a fresh machine (needs checkpoints/demo/expert_step.pt).
set -euo pipefail

expert="${EXPERT_CKPT:-checkpoints/demo/expert_step.pt}"
iters="${MAX_ITERATIONS:-1500}"
every="${EVAL_EVERY:-250}"
num_envs="${NUM_ENVS:-4096}"
seeds=(${SEEDS:-1 2 3})
configs=(${CONFIGS:-sched schedbc})
# Runs are assigned to these GPUs round-robin (e.g. GPUS="0 1" on Kaggle 2x T4).
gpus=(${GPUS:-0})
timing="${TIMING:-1}"
timing_iters="${TIMING_ITERS:-60}"
timing_configs=(${TIMING_CONFIGS:-${configs[*]}})
variant="${VARIANT:-}"
if [[ -n "$variant" ]]; then
  suffix="-${variant^}"         # e.g. -Payload
  tag="variant-$variant"        # run names and output files
  reference=(--reference-config ppo)
else
  suffix="" tag="phase2" reference=()
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
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

run_dir() { ls -td logs/rsl_rl/*/*_"$1" | head -1; }

echo "== expert data =="
collect Mjlab-Velocity-Flat-Unitree-G1-Expert "$base"
collect Mjlab-Velocity-Flat-Unitree-G1-Expert-Cond "$cond" --states-out "$states" \
  --bc-out "$bc"

echo "== training ($tag): ${configs[*]} x seeds ${seeds[*]}, $iters iterations =="
common=(MJLAB_AMP_EXPERT="$cond" MJLAB_AMP_STYLE_WEIGHT=4 MJLAB_AMP_TASK_LERP=0.3
  MJLAB_AMP_REPLAY=1000000 MJLAB_AMP_RSI_FILE="$states" MJLAB_AMP_RSI_PROB=0.85)
schedule=(MJLAB_AMP_LERP_START=0.9 MJLAB_AMP_LERP_HOLD=250 MJLAB_AMP_LERP_RAMP=500)
cloning=(MJLAB_AMP_BC_FILE="$bc" MJLAB_AMP_BC_STEPS=2000)
smooth=(MJLAB_AMP_ACTION_RATE=-0.3333)  # x 0.3 (task weight) = -0.1

config_task() {  # config_task <config>
  case "$1" in
    ppo) echo "Mjlab-Velocity-Flat-Unitree-G1-ExpertStep$suffix" ;;
    *) echo "Mjlab-Velocity-Flat-Unitree-G1-AMP-Cond$suffix" ;;
  esac
}

config_env() {  # config_env <config> -> sets the array "extra"
  case "$1" in
    sched) extra=("${common[@]}" "${schedule[@]}") ;;
    schedbc) extra=("${common[@]}" "${schedule[@]}" "${cloning[@]}") ;;
    v7c) extra=("${common[@]}") ;;
    bc) extra=("${common[@]}" "${cloning[@]}") ;;
    bcr) extra=("${common[@]}" "${cloning[@]}" "${smooth[@]}") ;;
    bcrs) extra=("${common[@]}" "${cloning[@]}" "${smooth[@]}" MJLAB_INIT_STD=0.3) ;;
    bcrsw) extra=("${common[@]}" "${cloning[@]}" "${smooth[@]}" MJLAB_INIT_STD=0.3
      MJLAB_AMP_CRITIC_WARMUP=25) ;;
    bctask) extra=("${common[@]}" "${cloning[@]}" MJLAB_AMP_STYLE_WEIGHT=0) ;;
    ppo) extra=(MJLAB_UNUSED=1) ;;  # stock runner: none of the AMP settings apply
    *) echo "unknown config: $1" >&2; exit 2 ;;
  esac
}

runs=()
n=0
for s in "${seeds[@]}"; do
  for c in "${configs[@]}"; do
    config_env "$c"
    gpu="${gpus[$((n % ${#gpus[@]}))]}"
    train "$(config_task "$c")" "$tag-$c-s$s" "$s" "${extra[@]}" \
      CUDA_VISIBLE_DEVICES="$gpu" &
    runs+=("$tag-$c-s$s")
    n=$((n + 1))
  done
done
wait

echo "== evaluation of every checkpoint =="
policies=(--policy "expert_step=$expert")
for r in "${runs[@]}"; do
  name="${r#"$tag"-}"        # e.g. bc-s1
  config="${name%-s*}"       # e.g. bc
  seed="${name##*-s}"
  d="$(run_dir "$r")"
  for ((it = every; it <= iters; it += every)); do
    ckpt="$d/model_$it.pt"
    [[ -f "$ckpt" ]] || ckpt="$d/model_$((it - 1)).pt"
    [[ -f "$ckpt" ]] && policies+=(--policy "${config}_s${seed}_it${it}=$ckpt")
  done
done
# On a variant, the nominal expert is evaluated there too (zero-shot transfer).
"${uv_run[@]}" python -m mjlab.tasks.velocity_amp.scripts.evaluate \
  --task "Mjlab-Velocity-Flat-Unitree-G1-Expert$suffix" \
  --expert-file "$base" "${policies[@]}" \
  --out "logs/amp_eval/results-$tag-$stamp.json" \
  > "logs/amp_console/eval-$tag-$stamp.log" 2>&1
"${uv_run[@]}" python -m mjlab.tasks.velocity_amp.scripts.convergence \
  "logs/amp_eval/results-$tag-$stamp.json" "${reference[@]}" \
  --out "logs/amp_eval/convergence-$tag-$stamp.json" | tee "logs/${tag}_summary.md"
# Same table with the stricter reward threshold (95% instead of 90%).
"${uv_run[@]}" python -m mjlab.tasks.velocity_amp.scripts.convergence \
  "logs/amp_eval/results-$tag-$stamp.json" "${reference[@]}" --min-reward-frac 0.95 \
  | tee -a "logs/${tag}_summary.md"

if [[ "$timing" == 1 ]]; then
  echo "== timing: each config alone on one GPU, $timing_iters iterations =="
  {
    echo
    echo "| config | seconds / iteration (alone on one GPU) |"
    echo "|---|---|"
    for c in "${timing_configs[@]}"; do
      config_env "$c"
      env "${extra[@]}" CUDA_VISIBLE_DEVICES="${gpus[0]}" "${uv_run[@]}" train \
        "$(config_task "$c")" --gpu-ids '[0]' --env.scene.num-envs "$num_envs" \
        --agent.max-iterations "$timing_iters" --agent.save-interval 100000 \
        --agent.logger tensorboard --agent.upload-model False \
        --agent.run-name "timing-$tag-$c" \
        > "logs/amp_console/timing-$c-$stamp.log" 2>&1
      # Median over the second half (the first iterations include compilation).
      t=$(sed 's/\x1b\[[0-9;]*m//g' "logs/amp_console/timing-$c-$stamp.log" \
        | grep -oE "Iteration time: [0-9.]+" | awk '{print $3}' \
        | tail -n $((timing_iters / 2)) | sort -n \
        | awk '{a[NR]=$1} END {if (NR) print a[int((NR + 1) / 2)]; else print "n/a"}')
      echo "| $c | $t |"
    done
  } | tee -a "logs/${tag}_summary.md"
fi

# Small bundle for a slow link: results, summary, logs, TensorBoard events and
# the final checkpoint of each run.
files=("logs/amp_eval/results-$tag-$stamp.json" "logs/amp_eval/convergence-$tag-$stamp.json"
  "logs/${tag}_summary.md" logs/amp_console/*-"$stamp".log)
for r in "${runs[@]}"; do
  d="$(run_dir "$r")"
  files+=("$d"/events.out.* "$(ls -t "$d"/model_*.pt | head -1)")
  # Also every 250th checkpoint, for demos of earlier stages.
  for ((it = 250; it < iters; it += 250)); do
    [[ -f "$d/model_$it.pt" ]] && files+=("$d/model_$it.pt")
  done
done
tar czf "logs/${tag}_results-$stamp.tgz" "${files[@]}"
ls -la "logs/${tag}_results-$stamp.tgz"
echo "Done $tag."
