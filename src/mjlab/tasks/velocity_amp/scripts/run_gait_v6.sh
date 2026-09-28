#!/usr/bin/env bash
# Round 6 (after run_gait_v5.sh, on the same machine):
#   v6c = resume GAIL v5c (stepping expert, key-body features) for 3000 more
#         iterations; its style reward was still rising at 2000.
#   v6d = stepping expert + lite gait features (foot height, 5-frame window,
#         no contact flags) so the discriminator does not saturate like v5a.
set -euo pipefail

extra="${EXTRA_ITERATIONS:-3000}"
iters_d="${ITERS_D:-4000}"
num_envs="${NUM_ENVS:-4096}"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MJLAB_INIT_STD=1.0 MJLAB_ENTROPY_COEF=0.0 MJLAB_AMP_LOSS=gail
export MJLAB_AMP_STYLE_WEIGHT=2 MJLAB_AMP_DISC_UPDATES=5 MJLAB_AMP_DISC_LR=1e-4
uv_run=(uv run --frozen --no-dev --extra cu128)
stamp="$(date +%Y%m%d-%H%M%S)"

new_expert="$(ls -td logs/rsl_rl/g1_velocity_expert_step/*_expert-step | head -1)/model_2999.pt"
v5c_dir="$(ls -td logs/rsl_rl/g1_velocity_amp/*_gail-v5c | head -1)"
lite=logs/amp_expert/expert_step_gaitlite.npz

common=(--gpu-ids '[0]' --env.scene.num-envs "$num_envs" --agent.save-interval 250
  --agent.logger tensorboard --agent.upload-model False)

echo "== data =="
[[ -f "$lite" ]] || "${uv_run[@]}" python -m mjlab.tasks.velocity_amp.scripts.collect_expert \
  "$new_expert" --task Mjlab-Velocity-Flat-Unitree-G1-Expert-GaitLite --out "$lite" \
  --num-envs 512 --steps 250 > "logs/amp_console/collect-gaitlite-$stamp.log" 2>&1

echo "== training (parallel) =="
MJLAB_AMP_EXPERT=logs/amp_expert/expert_step_kb.npz "${uv_run[@]}" train \
  Mjlab-Velocity-Flat-Unitree-G1-AMP-KeyBody "${common[@]}" \
  --agent.max-iterations "$extra" --agent.run-name gail-v6c \
  --agent.resume True --agent.load-run "$(basename "$v5c_dir")" \
  --agent.load-checkpoint model_1999.pt > "logs/amp_console/gail-v6c-$stamp.log" 2>&1 &
MJLAB_AMP_EXPERT="$lite" "${uv_run[@]}" train \
  Mjlab-Velocity-Flat-Unitree-G1-AMP-GaitLite "${common[@]}" \
  --agent.max-iterations "$iters_d" --agent.run-name gail-v6d \
  > "logs/amp_console/gail-v6d-$stamp.log" 2>&1 &
wait

last() { ls -t "$(ls -td logs/rsl_rl/g1_velocity_amp/*_"$1" | head -1)"/model_*.pt | head -1; }
v6c="$(last gail-v6c)"; v6d="$(last gail-v6d)"
echo "v6c: $v6c"; echo "v6d: $v6d"

echo "== evaluation =="
"${uv_run[@]}" python -m mjlab.tasks.velocity_amp.scripts.evaluate \
  --expert-file logs/amp_expert/g1_velocity_expert.npz \
  --policy "expert_step=$new_expert" \
  --policy "gail_v4b=checkpoints/demo/gail_v4b.pt" \
  --policy "gail_v5c=$v5c_dir/model_1999.pt" \
  --policy "gail_v6c=$v6c" --policy "gail_v6d=$v6d" \
  --out "logs/amp_eval/results-v6-$stamp.json" > "logs/amp_console/eval-v6-$stamp.log" 2>&1
grep -E '^\| (policy|expert|gail)' "logs/amp_console/eval-v6-$stamp.log"

echo "== snapshots =="
"${uv_run[@]}" python report/render_snapshots.py --out report/figures \
  --policy "gail_v6c=$v6c" --policy "gail_v6d=$v6d" \
  > "logs/amp_console/snap-v6-$stamp.log" 2>&1 || echo "snapshots failed (see log)"
echo "Done v6."
