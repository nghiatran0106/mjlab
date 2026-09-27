#!/usr/bin/env bash
# Round 5: fix the shuffling gait of GAIL v4b.
#   GAIL v4b takes ~7 steps/s per foot with 1.3 cm foot lift (expert: ~3 steps/s,
#   2.7 cm). Two causes, tested separately:
#   (1) the expert itself lifts its feet little -> train a stepping expert
#       (air-time reward on, stronger swing-height penalty), 3000 iterations;
#   (2) the discriminator sees a single 20 ms transition without foot contact
#       -> "gait" features: 0.24 s window + foot height/contact.
#   v5a = new expert + gait features   (both fixes)
#   v5b = old expert + gait features   (fix 2 only; runs while the expert trains)
#   v5c = new expert + key-body only   (fix 1 only, same features as v4b)
# All GAIL runs use the v4b settings. Works on a fresh machine.
set -euo pipefail

old_expert="${EXPERT_CKPT:-checkpoints/demo/expert.pt}"
v4b_ckpt="${V4B_CKPT:-checkpoints/demo/gail_v4b.pt}"
num_envs="${NUM_ENVS:-4096}"
expert_iters="${EXPERT_ITERS:-3000}"
iters="${MAX_ITERATIONS:-2000}"

export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
uv_run=(uv run --frozen --no-dev --extra cu128)
mkdir -p logs/amp_console logs/amp_expert logs/amp_eval
stamp="$(date +%Y%m%d-%H%M%S)"

collect() {  # collect <ckpt> <task> <out>
  [[ -f "$3" ]] && return
  "${uv_run[@]}" python -m mjlab.tasks.velocity_amp.scripts.collect_expert \
    "$1" --task "$2" --out "$3" --num-envs 512 --steps 250 \
    > "logs/amp_console/collect-$(basename "$3" .npz)-$stamp.log" 2>&1
}

train() {  # train <task> <run-name> <iterations>
  "${uv_run[@]}" train "$1" \
    --gpu-ids '[0]' \
    --env.scene.num-envs "$num_envs" \
    --agent.max-iterations "$3" \
    --agent.save-interval 250 \
    --agent.logger tensorboard \
    --agent.upload-model False \
    --agent.run-name "$2" > "logs/amp_console/$2-$stamp.log" 2>&1
}

gail() {  # gail <task> <expert-npz> <run-name>
  MJLAB_INIT_STD=1.0 MJLAB_ENTROPY_COEF=0.0 MJLAB_AMP_LOSS=gail \
    MJLAB_AMP_STYLE_WEIGHT=2 MJLAB_AMP_DISC_UPDATES=5 MJLAB_AMP_DISC_LR=1e-4 \
    MJLAB_AMP_EXPERT="$2" train "$1" "$3" "$iters"
}

latest() {  # latest <experiment> <run-name> <iterations>
  echo "$(ls -td logs/rsl_rl/"$1"/*_"$2" | head -1)/model_$(($3 - 1)).pt"
}

base=logs/amp_expert/g1_velocity_expert.npz
old_gait=logs/amp_expert/expert_old_gait.npz
new_gait=logs/amp_expert/expert_step_gait.npz
new_kb=logs/amp_expert/expert_step_kb.npz

echo "== phase A: stepping expert + v5b (parallel) =="
collect "$old_expert" Mjlab-Velocity-Flat-Unitree-G1-Expert "$base"
collect "$old_expert" Mjlab-Velocity-Flat-Unitree-G1-Expert-Gait "$old_gait"
train Mjlab-Velocity-Flat-Unitree-G1-ExpertStep expert-step "$expert_iters" &
pid_expert=$!
gail Mjlab-Velocity-Flat-Unitree-G1-AMP-Gait "$old_gait" gail-v5b &
pid_v5b=$!
wait "$pid_expert"
new_expert="$(latest g1_velocity_expert_step expert-step "$expert_iters")"
echo "new expert: $new_expert"

echo "== phase B: v5a + v5c (parallel) =="
collect "$new_expert" Mjlab-Velocity-Flat-Unitree-G1-Expert-Gait "$new_gait"
collect "$new_expert" Mjlab-Velocity-Flat-Unitree-G1-Expert-KeyBody "$new_kb"
gail Mjlab-Velocity-Flat-Unitree-G1-AMP-Gait "$new_gait" gail-v5a &
gail Mjlab-Velocity-Flat-Unitree-G1-AMP-KeyBody "$new_kb" gail-v5c &
wait
wait "$pid_v5b"

echo "== evaluation =="
v5() { latest g1_velocity_amp "gail-v5$1" "$iters"; }
"${uv_run[@]}" python -m mjlab.tasks.velocity_amp.scripts.evaluate \
  --expert-file "$base" \
  --policy "expert=$old_expert" \
  --policy "expert_step=$new_expert" \
  --policy "gail_v4b=$v4b_ckpt" \
  --policy "gail_v5a=$(v5 a)" \
  --policy "gail_v5b=$(v5 b)" \
  --policy "gail_v5c=$(v5 c)" \
  --out "logs/amp_eval/results-v5-$stamp.json" \
  > "logs/amp_console/eval-v5-$stamp.log" 2>&1
grep -E '^\| ' "logs/amp_console/eval-v5-$stamp.log"

echo "== snapshots =="
"${uv_run[@]}" python report/render_snapshots.py --out report/figures \
  --policy "expert_step=$new_expert" --policy "gail_v5a=$(v5 a)" \
  --policy "gail_v5b=$(v5 b)" --policy "gail_v5c=$(v5 c)" \
  > "logs/amp_console/snap-v5-$stamp.log" 2>&1 || echo "snapshots failed (see log)"
echo "Done v5."
