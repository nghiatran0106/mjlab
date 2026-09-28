#!/usr/bin/env bash
# One evaluation for every policy, with identical settings, so versions are
# comparable: same script, same 512 envs x 20 s, same seed and command
# distribution, no pushes, deterministic policy, same true-reward terms and the
# same reference expert for the distribution gap.
#
# Waits for round 7 to reach STOP_ITER, stops its training (the goal is
# convergence within 2000 iterations), then evaluates round-7 checkpoints at
# 500/1000/1500/2000 plus the final checkpoint of every earlier round
# (checkpoints/all/*.pt), renders snapshots and packs a small bundle.
set -euo pipefail

stop_iter="${STOP_ITER:-2000}"
ref=logs/amp_expert/v7_expert_base.npz  # stepping expert, 68-d base features
runs=(gail-v7a gail-v7b gail-v7c amp-v7d)
uv_run=(uv run --frozen --no-dev --extra cu128)
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
stamp="$(date +%Y%m%d-%H%M%S)"

run_dir() { ls -td logs/rsl_rl/g1_velocity_amp/*_"$1" | head -1; }

echo "== waiting for round 7 to reach iteration $stop_iter =="
for r in "${runs[@]}"; do
  until [[ -f "$(run_dir "$r")/model_$stop_iter.pt" ]]; do sleep 30; done
done
sleep 30  # let the last checkpoint finish writing
tmux kill-session -t v7 2>/dev/null || true
pkill -f "train Mjlab-Velocity" 2>/dev/null || true
echo "stopped round 7 at iteration $stop_iter"

policies=()
for f in checkpoints/all/*.pt; do
  policies+=(--policy "$(basename "$f" .pt)=$f")
done
for r in "${runs[@]}"; do
  name="${r//-/_}"
  for it in 500 1000 1500 "$stop_iter"; do
    policies+=(--policy "${name}_it${it}=$(run_dir "$r")/model_$it.pt")
  done
done

echo "== evaluation (${#policies[@]} args) =="
"${uv_run[@]}" python -m mjlab.tasks.velocity_amp.scripts.evaluate \
  --expert-file "$ref" "${policies[@]}" \
  --out "logs/amp_eval/results-fair-$stamp.json" \
  > "logs/amp_console/eval-fair-$stamp.log" 2>&1 &
pid_eval=$!

echo "== snapshots (CPU, in parallel) =="
snaps=()
for f in checkpoints/all/*.pt; do snaps+=(--policy "$(basename "$f" .pt)=$f"); done
for r in "${runs[@]}"; do
  snaps+=(--policy "${r//-/_}_it${stop_iter}=$(run_dir "$r")/model_$stop_iter.pt")
done
"${uv_run[@]}" python report/render_snapshots.py --out report/figures "${snaps[@]}" \
  > "logs/amp_console/snap-fair-$stamp.log" 2>&1 || echo "snapshots failed (see log)"
wait "$pid_eval"
grep -E '^\| ' "logs/amp_console/eval-fair-$stamp.log" | grep -vE "Index|Active|Property|Name|  [0-9]+  " || true

bundle="logs/fair_results-$stamp.tgz"
files=("logs/amp_eval/results-fair-$stamp.json" "logs/amp_console/eval-fair-$stamp.log")
files+=(report/figures/snap_*.png report/figures/snapshots.json)
for r in "${runs[@]}"; do
  d="$(run_dir "$r")"
  files+=("$d/model_$stop_iter.pt" "$d"/events.out.* "logs/amp_console/$r"-*.log)
done
tar czf "$bundle" "${files[@]}"
ls -la "$bundle"
echo "Done fair eval."
