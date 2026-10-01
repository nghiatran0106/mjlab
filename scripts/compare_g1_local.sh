#!/usr/bin/env bash
# So sánh nhiều policy cùng lúc trong một viewer (CPU), mở http://localhost:8080.
#   bash scripts/compare_g1_local.sh            # bộ "main": expert, task-only, GAIL v1, v4b, v7c
#   bash scripts/compare_g1_local.sh phase1     # expert, v7c seed 1/2/3, chỉ RSI
#   bash scripts/compare_g1_local.sh phase2     # expert, v7c, bc seed 1/2/3
#   bash scripts/compare_g1_local.sh final      # expert, PPO gốc, GAIL+BC, BC không GAIL (3 seed mỗi loại), 3 khung
#   bash scripts/compare_g1_local.sh v3         # báo cáo v3: expert, PPO gốc, GAIL+BC cải tiến (bcrs1), GAIL+BC, BC
#   bash scripts/compare_g1_local.sh all        # nạp mọi checkpoint, 2 khung, chọn policy trong menu
#   bash scripts/compare_g1_local.sh custom --policy a=path.pt --policy b=path.pt
# Mỗi khung có menu "Policy" để đổi sang bất kỳ policy nào đã nạp; --views 2 = số khung.
# CPU mô phỏng chậm hơn thời gian thực (~0.2-0.4x). --record 20: mô phỏng trước 20 s
#   rồi phát lại mượt, chỉnh tốc độ (Slower/1x/Faster) và tua bằng thanh Time.
# Thêm tham số: --lin-x 1.0 (tốc độ lệnh), --yaw 0.5 (xoay), --random-commands,
#   --columns 2 (số khung mỗi hàng), --single (tất cả robot trong một khung)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

D=checkpoints/demo
P1=phase1_results-20260928-091927/logs/rsl_rl/g1_velocity_amp
preset="${1:-main}"
[[ $# -gt 0 ]] && shift
case "$preset" in
  main) policies=(--policy "expert=$D/expert_step.pt" --policy "task_only=$D/task_only.pt"
          --policy "gail_v1=$D/gail_v1.pt" --policy "gail_v4b=$D/gail_v4b.pt"
          --policy "gail_v7c=$D/gail_v7c.pt") ;;
  phase1) policies=(--policy "expert=$D/expert_step.pt"
          --policy "v7c_s1=$(ls $P1/*_p1-v7c-s1/model_*.pt | tail -1)"
          --policy "v7c_s2=$(ls $P1/*_p1-v7c-s2/model_*.pt | tail -1)"
          --policy "v7c_s3=$(ls $P1/*_p1-v7c-s3/model_*.pt | tail -1)"
          --policy "rsi_only=$(ls $P1/*_p1-rsionly-s1/model_*.pt | tail -1)") ;;
  phase2) B=phase2_results-20260929-073946/logs/rsl_rl/g1_velocity_amp
          policies=(--policy "expert=$D/expert_step.pt" --policy "v7c=$D/gail_v7c.pt"
          --policy "bc_s1=$(ls $B/*_p2-bc-s1/model_*.pt | tail -1)"
          --policy "bc_s2=$(ls $B/*_p2-bc-s2/model_*.pt | tail -1)"
          --policy "bc_s3=$(ls $B/*_p2-bc-s3/model_*.pt | tail -1)") ;;
  all) B=phase2_results-20260929-073946/logs/rsl_rl/g1_velocity_amp
       policies=(--views 2)
       for f in $D/*.pt; do policies+=(--policy "$(basename "$f" .pt)=$f"); done
       for s in 1 2 3; do
         policies+=(--policy "v7c_s$s=$(ls $P1/*_p1-v7c-s$s/model_*.pt | tail -1)")
         policies+=(--policy "bc_s$s=$(ls $B/*_p2-bc-s$s/model_*.pt | tail -1)")
       done ;;
  final) B=phase2_results-20260929-073946/logs/rsl_rl/g1_velocity_amp
         C=phase2_results-20260930-020842/logs/rsl_rl
         policies=(--views 3 --policy "expert=$D/expert_step.pt")
         for s in 1 2 3; do
           policies+=(--policy "ppo_s$s=$(ls $C/g1_velocity_expert_step/*_p2-ppo-s$s/model_*.pt | tail -1)")
           policies+=(--policy "gail_bc_s$s=$(ls $B/*_p2-bc-s$s/model_*.pt | tail -1)")
           policies+=(--policy "bc_nogail_s$s=$(ls $C/g1_velocity_amp/*_p2-bctask-s$s/model_*.pt | tail -1)")
         done ;;
  v3) A=phase2_results-20260930-020842/logs/rsl_rl
      B=phase2_results-20260929-073946/logs/rsl_rl/g1_velocity_amp
      N=phase2_results-20260930-223212/logs/rsl_rl/g1_velocity_amp
      policies=(--views 3 --policy "expert=$D/expert_step.pt"
        --policy "ppo_s1_it1500=$(ls $A/g1_velocity_expert_step/*_p2-ppo-s1/model_*.pt | tail -1)"
        --policy "gail_bc_cai_tien_s1_it750=$N/$(ls $N | grep bcrs1-s1)/model_749.pt"
        --policy "gail_bc_cai_tien_s1_it250=$N/$(ls $N | grep bcrs1-s1)/model_250.pt"
        --policy "gail_bc_cai_tien_s2_it750=$N/$(ls $N | grep bcrs1-s2)/model_749.pt"
        --policy "gail_bc_cu_s1_it1500=$(ls $B/*_p2-bc-s1/model_*.pt | tail -1)"
        --policy "bc_khong_gail_s1_it1500=$(ls $A/g1_velocity_amp/*_p2-bctask-s1/model_*.pt | tail -1)") ;;
  custom) policies=() ;;
  *) echo "Preset: main | phase1 | phase2 | final | v3 | all | custom" >&2; exit 2 ;;
esac

local_python="${MJLAB_PYTHON:-$HOME/venvs/mjlab/bin/python}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export MUJOCO_GL=disable CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/mjlab-uv-cache}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/mjlab-warp-cache}"
exec "$local_python" -m mjlab.tasks.velocity_amp.scripts.compare \
  "${policies[@]}" --device cpu "$@"
