#!/usr/bin/env bash
# Phase 2 presets on one Colab GPU (Colab CLI: `colab new --gpu A100`), detached.
#   bash colab_phase2.sh setup            # clone research/amp-velocity, uv sync
#   bash colab_phase2.sh bench            # GPU info + seconds/iteration, 1 and 4 runs
#   bash colab_phase2.sh start "A C"      # run presets one after another, detached
#   bash colab_phase2.sh status           # progress of the detached job
# Presets match notebooks/kaggle_phase2.ipynb: A/B fair reruns on the nominal robot,
# C/D reuse on the weak/slippery robot; seeds 4 5 6, 1500 iterations, eval every 50.
# All runs of a preset share the GPU (run_phase2.sh starts them together); the
# per-config timing still runs each config alone, so seconds/iteration stay fair.
set -euo pipefail

self="$(readlink -f "$0")"
repo=/content/mjlab
work=/content/phase2
uv_run=(uv run --frozen --no-dev --extra cu128)

preset_env() {  # preset_env <A|B|C|D> -> prints env assignments
  case "$1" in
    A) echo 'VARIANT= TAG=fair CONFIGS="ppo bcrs1 bcrsl" TIMING_CONFIGS="ppo bcrs1 bcrsl"' ;;
    B) echo 'VARIANT= TAG=fair CONFIGS="bctask v7c bc" TIMING_CONFIGS="ppo bctask v7c bc"' ;;
    C) echo 'VARIANT=weak TAG=variant-weak CONFIGS="ppo ft bcrs1" TIMING_CONFIGS="ppo ft bcrs1"' ;;
    D) echo 'VARIANT=slippery TAG=variant-slippery CONFIGS="ppo ft bcrs1"
      TIMING_CONFIGS="ppo ft bcrs1"' ;;
    *) echo "unknown preset $1" >&2; exit 2 ;;
  esac
}

[[ "${1:-}" == env ]] || mkdir -p "$work"

case "${1:-}" in
  setup)
    command -v uv >/dev/null || pip install -q uv
    rm -rf "$repo"
    git clone -q --depth 1 -b research/amp-velocity \
      https://github.com/nghiatran0106/mjlab.git "$repo"
    cd "$repo" && git log --oneline -1
    UV_LINK_MODE=copy uv sync --locked --no-dev --extra cu128 > "$work/uv_sync.log" 2>&1 \
      || { tail -20 "$work/uv_sync.log"; exit 1; }
    "${uv_run[@]}" python -c "import torch, warp as wp; wp.init(); \
print(torch.cuda.get_device_name(0), 'warp CUDA', wp.is_cuda_available())"
    ;;
  bench)
    cd "$repo"
    nvidia-smi --query-gpu=name,memory.total --format=csv
    export MUJOCO_GL=egl
    bench() {  # bench <parallel runs>
      for ((k = 0; k < $1; k++)); do
        "${uv_run[@]}" train Mjlab-Velocity-Flat-Unitree-G1-ExpertStep --gpu-ids '[0]' \
          --env.scene.num-envs 4096 --agent.max-iterations 30 --agent.save-interval 1000 \
          --agent.logger tensorboard --agent.upload-model False \
          --agent.run-name "bench$1-$k" > "$work/bench$1-$k.log" 2>&1 &
      done
      sleep 120; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
      wait
      for ((k = 0; k < $1; k++)); do
        sed 's/\x1b\[[0-9;]*m//g' "$work/bench$1-$k.log" | grep -oE "Iteration time: [0-9.]+" \
          | tail -10 | awk -v n="$1" '{s += $3} END {printf "%d runs: %.2f s/it\n", n, s / NR}'
      done
    }
    bench 1
    bench 4
    ;;
  start)
    presets="${2:?presets, e.g. \"A C\"}"
    cat > "$work/job.sh" <<EOF
set -uo pipefail
cd $repo
for p in $presets; do
  echo "== preset \$p \$(date)"
  eval "\$(bash $self env \$p)"
  env VARIANT="\$VARIANT" TAG="\$TAG" CONFIGS="\$CONFIGS" TIMING_CONFIGS="\$TIMING_CONFIGS" \
    SEEDS="4 5 6" MAX_ITERATIONS=1500 EVAL_EVERY=50 GPUS=0 TIMING=1 \
    EVAL_JOBS="\${EVAL_JOBS:-6}" MUJOCO_GL=egl \
    bash src/mjlab/tasks/velocity_amp/scripts/run_phase2.sh
  cp logs/*_results-*.tgz logs/*_summary.md $work/ 2>/dev/null
done
echo "== all done \$(date)"
EOF
    nohup bash "$work/job.sh" > "$work/job.log" 2>&1 &
    echo "started pid $! (log $work/job.log)"
    ;;
  env) preset_env "$2" | tr '\n' ' ' | sed 's/^/export /' ;;
  status)
    tail -5 "$work/job.log" 2>/dev/null || echo "no job"
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
    for f in "$repo"/logs/amp_console/*-s[0-9]-*.log; do
      [[ -f "$f" ]] || continue
      it=$(sed 's/\x1b\[[0-9;]*m//g' "$f" | grep -oE "Learning iteration [0-9]+/[0-9]+" | tail -1)
      echo "$(basename "$f" .log): ${it:-starting}"
    done
    ls -la "$work"/*.tgz 2>/dev/null || true
    ;;
  *) sed -n '2,9p' "$0"; exit 2 ;;
esac
