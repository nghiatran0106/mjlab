# Training Unitree G1 trên Vast.ai

Task đã chọn: `Mjlab-Velocity-Flat-Unitree-G1`.
Repo được khảo sát tại commit `8ee51fbcf` (mjlab 1.6.0).

## Repo này làm gì?

Đây là reinforcement learning cho robot trong mô phỏng. Với task velocity,
không cần tải dataset: robot thu thập rollout bằng tương tác với môi trường.

| Thành phần | Vai trò |
| --- | --- |
| `src/mjlab/asset_zoo`, `entity`, `actuator` | Mô hình robot, khớp và động cơ |
| `scene`, `sim`, `sensor`, `terrains` | Tạo cảnh MuJoCo và mô phỏng bằng Warp |
| `envs`, `managers` | Observation, action, reward, reset, termination, curriculum |
| `tasks/velocity/config/g1` | Cấu hình G1 flat/rough và PPO |
| `tasks/tracking`, `manipulation`, `cartpole` | Các nhóm bài toán khác |
| `rl`, `scripts/train.py` | Thu rollout, cập nhật PPO, checkpoint, export ONNX |
| `scripts/play.py`, `viewer` | Xem lại policy bằng MuJoCo hoặc trình duyệt |
| `tests`, `docs`, `scripts/benchmarks` | Kiểm thử, tài liệu, đo hiệu năng |

Luồng chính: task registry → scene/robot → môi trường có các manager →
`RslRlVecEnvWrapper` → PPO actor/critic → checkpoint → play/evaluation.
G1 có actor và critic MLP `(512, 256, 128)`, rollout 24 bước/môi trường,
5 epoch PPO, 4 minibatch. Cấu hình gốc chạy **30.000 iteration**.
Script đi kèm chọn **1.000 iteration cho lần thử đầu**, không phải cam kết hội tụ.
Curriculum vận tốc còn thay đổi ở mốc 5.000 và 10.000 iteration.

Dependencies lấy từ `uv.lock`; lựa chọn `cu128` dùng PyTorch CUDA 12.8.
Tất cả lệnh Python bên dưới đi qua `uv run` theo `CLAUDE.md`.

## 1. Đăng ký và nạp tiền

1. Mở <https://cloud.vast.ai/>, tạo tài khoản và xác minh email.
2. Vào **Billing → Add Credits**, thêm phương thức thanh toán thẻ qua Stripe.
   Nhập thông tin thẻ trực tiếp trong form thanh toán, không gửi qua chat.
3. Đề xuất nạp **5 USD** nếu chấp nhận tăng tiền nạp ban đầu. Theo quickstart,
   mức nạp tối thiểu là **5 USD**; đây không phải giá đảm bảo cho cả project.
4. Đọc thiết lập **Autobilling**, số tiền tự nạp và thông báo số dư thấp.
   Số tiền nạp ban đầu không phải một giới hạn chi tiêu cứng.

Vast tính riêng GPU, dung lượng lưu trữ và lưu lượng mạng. **Stop vẫn tính
phí lưu trữ**; thẻ đã lưu có thể bị trừ tiền để bù số dư âm. Sao lưu trước,
rồi **Destroy/Delete** instance khi không còn cần. Đừng chờ hết credit để
dừng chi phí: dữ liệu có thể bị xóa.

Nguồn chính thức, kiểm tra ngày 20/09/2026:
[Quickstart](https://docs.vast.ai/guides/get-started/quickstart),
[Billing](https://docs.vast.ai/guides/reference/billing).

### Ngân sách của bạn: 50.000 đồng

Nếu tài khoản chưa có credit, 50.000 đồng chưa đủ mức nạp tối thiểu 5 USD.
Với tỷ giá tham khảo khoảng 26.000 VND/USD, cần khoảng **130.000 đồng**
cộng phí chuyển đổi của ngân hàng để nạp 5 USD. Số tiền chính xác theo thẻ
tại thời điểm thanh toán. Nguồn tỷ giá tham khảo:
[USD/VND](https://vn.investing.com/currencies/usd-vnd-historical-data).

Đề xuất giữ mục tiêu **job tiêu thụ tối đa khoảng 1,50 USD**, chừa phần còn
lại trong ngân sách 50.000 đồng cho setup, truyền dữ liệu và chênh lệch tỷ giá.
Credit chưa dùng vẫn ở tài khoản. Đây là mục tiêu vận hành, không phải
cơ chế tự động giới hạn hóa đơn của script.

Sau benchmark, chỉ chạy 1.000 iteration trước để xem policy/đường học:

```bash
MAX_ITERATIONS=1000 bash scripts/cloud/train_vast.sh train
```

Sau đó resume đến 6.000 hoặc hơn nếu dự toán còn trong ngân sách. So sánh
offer bằng `giá tổng/giờ × giây/iteration`; không chỉ dùng giá GPU/giờ.
Ví dụ giả định 3090 giá 0,20 USD/giờ chạy 2 giây/iteration sẽ đắt hơn
4090 giá 0,35 USD/giờ chạy 1 giây/iteration cho cùng số iteration.
Không thuê nhiều máy chỉ để benchmark với ngân sách nhỏ; bắt đầu bằng một
offer phù hợp, chỉ đổi nếu hiệu năng thực tế quá thấp. Không thể đảm bảo
robot hội tụ trong 50.000 đồng trước khi có số đo GPU.

## 2. Chọn GPU

Đề xuất khởi đầu cho cấu hình này, cần xác nhận bằng benchmark thực tế:

| Mục | Đề xuất |
| --- | --- |
| GPU | 1 × RTX 4090 24 GB; so sánh RTX 3090 24 GB nếu rẻ hơn |
| RAM cấp cho instance | Từ 32 GB |
| CPU cấp cho instance | Từ 8 vCPU |
| Disk | 50–80 GB, tùy thời gian giữ checkpoint/video |
| Kiểu thuê | On-demand cho lần đầu |
| Host | Verified, reliability cao, mạng đủ nhanh để tải dependencies |
| Template | PyTorch, kết nối SSH, môi trường Linux hỗ trợ CUDA 12.8 |

Đây là cấu hình thử đề xuất, không phải mức tối thiểu đã được đo trên GPU.
PyTorch dùng CUDA 12.8, còn Warp 1.14.0 trong môi trường đã kiểm tra báo
CUDA Toolkit 12.9. Để thuận tiện, ưu tiên driver host hỗ trợ CUDA 12.9 trở
lên bằng `nvidia-smi`; image CUDA không tự nâng cấp driver host. Script
kiểm tra cả PyTorch và Warp trước khi train. Không cần nhiều GPU cho lần đầu.
Nếu thiếu VRAM, giảm `NUM_ENVS=2048` hoặc `1024` và benchmark lại.

Trên **Search/Create**, xem tổng giá và phí mạng bằng chi tiết offer trước
khi bấm **Rent**. Giá thay đổi theo host/thời điểm, chưa có báo giá trực tiếp
cho tài khoản của bạn. Disk của instance không thể đổi kích thước sau khi tạo.
Nguồn: [chọn instance](https://docs.vast.ai/guides/instances/choosing/find-and-rent),
[quickstart](https://docs.vast.ai/guides/get-started/quickstart).

## 3. SSH và chuyển đúng repo hiện tại

Tạo key riêng trên máy cá nhân nếu chưa có (không ghi đè key cũ):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/mjlab_vast -C mjlab-vast
cat ~/.ssh/mjlab_vast.pub
```

Thêm nội dung `.pub` vào **Keys** của Vast trước khi tạo instance.
Sau khi instance sẵn sàng, lấy host/port/user trong nút **Connect/Open**.
Chỉ public key được tải lên; private key giữ trên máy cá nhân.
Nguồn: [SSH của Vast](https://docs.vast.ai/guides/instances/connect/ssh).

Các lệnh sau dùng `HOST`, `PORT`, `USER` làm chỗ điền thông tin Connect:

```bash
ssh -i ~/.ssh/mjlab_vast -p PORT USER@HOST 'mkdir -p /workspace/mjlab'

# Chạy từ thư mục repo trên máy cá nhân; giữ nguyên slash cuối nguồn.
rsync -rt --info=progress2 \
  -e 'ssh -i ~/.ssh/mjlab_vast -p PORT' \
  --exclude='.venv/' --exclude='logs/' --exclude='wandb/' \
  --exclude='__pycache__/' \
  ./ USER@HOST:/workspace/mjlab/

ssh -i ~/.ssh/mjlab_vast -p PORT USER@HOST
```

Việc chuyển repo bao gồm cả script mới và đúng source local; không dùng
`git clone main` để rồi vô tình chạy một phiên bản khác.

## 4. Cài trong instance rồi chạy pipeline

Trong terminal SSH của instance (các lệnh apt dành cho container chạy root;
nếu là user thường thì dùng sudo theo quyền của instance):

```bash
apt-get update
apt-get install -y git curl ca-certificates libegl1 libgl1 tmux rsync
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd /workspace/mjlab
uv sync --locked --no-dev --extra cu128
uv run --frozen --no-dev --extra cu128 list-envs
nvidia-smi

# 128 env, 10 iteration: kiểm tra CUDA, simulation, PPO, lưu model.
bash scripts/cloud/train_vast.sh smoke

# 4096 env, 50 iteration: đo cả rollout và PPO update.
bash scripts/cloud/train_vast.sh benchmark
```

Chỉ tiếp tục khi smoke/benchmark chạy hết và có `model_*.pt` cùng file ONNX.
Script dùng TensorBoard local nên không cần W&B API key.
Lần đầu Warp biên dịch kernel có thể mất nhiều phút; xem log trước khi kết
luận chương trình bị treo.

Lấy trung bình `Iteration time` ở các iteration cuối của benchmark (bỏ
warmup/compilation). Ước lượng thời gian còn lại:

```text
giờ ≈ số iteration × giây/iteration / 3600
chi phí ≈ giờ × (giá GPU/giờ + disk/giờ) + phí truyền dữ liệu
```

Ví dụ giả định 1 giây/iteration và tổng 0,40 USD/giờ thì 6.000 iteration
tốn khoảng 1,67 giờ và 0,67 USD, chưa gồm setup/mạng. Đây chỉ là phép tính
minh họa, không phải benchmark hay báo giá. `measure_throughput.py` chỉ đo
môi trường, không bao gồm cập nhật PPO nên không đủ để dự toán toàn bộ job.

Chạy dài trong tmux để giữ job khi mất SSH:

```bash
tmux new -s mjlab-g1
cd /workspace/mjlab
MAX_ITERATIONS=1000 bash scripts/cloud/train_vast.sh train
# Ctrl-b rồi d: rời tmux. Kết nối lại: tmux attach -t mjlab-g1
```

Script mặc định 1.000 iteration nếu bỏ `MAX_ITERATIONS`. Để chạy theo
số iteration gốc:

```bash
MAX_ITERATIONS=30000 bash scripts/cloud/train_vast.sh train
```

Không tự động chạy lệnh 30.000 iteration nếu chưa tính đủ ngân sách.
Training kết thúc **không tự Stop/Destroy instance**.

## 5. Kết quả, xem policy và resume

Kết quả nằm trong:

```text
logs/rsl_rl/g1_velocity/<timestamp>_vast-train/
  model_0.pt, model_50.pt, ...
  <run-directory-name>.onnx
  params/env.yaml
  params/agent.yaml
  events.out.tfevents.*
logs/vast/train-*.log
```

Xem reward, episode length và lỗi tracking trên TensorBoard. Cần kiểm tra
policy thực tế; việc tạo được checkpoint không chứng minh robot đã đi tốt.

```bash
uv run --frozen --no-dev --extra cu128 tensorboard \
  --logdir logs/rsl_rl/g1_velocity --host 127.0.0.1 --port 6006
```

Mở terminal khác trên máy cá nhân, dùng SSH tunnel:

```bash
ssh -i ~/.ssh/mjlab_vast -p PORT \
  -L 6006:127.0.0.1:6006 -L 8080:127.0.0.1:8080 USER@HOST
```

Mở `http://localhost:6006`. Xem policy từ terminal SSH khác:

```bash
uv run --frozen --no-dev --extra cu128 play Mjlab-Velocity-Flat-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_velocity/RUN_DIR/model_5999.pt \
  --num-envs 1 --viewer viser
```

Thay `RUN_DIR` và tên checkpoint bằng file thực tế; mở địa chỉ viewer
(mặc định `http://localhost:8080`). Dừng viewer bằng Ctrl-C.

Resume chính xác một run thay vì vô tình lấy run smoke/benchmark mới nhất:

```bash
MAX_ITERATIONS=3000 bash scripts/cloud/train_vast.sh train \
  --agent.resume True \
  --agent.load-run 'TEN_THU_MUC_RUN_CU' \
  --agent.load-checkpoint 'model_5999.pt'
```

`max-iterations` lúc resume là số iteration **chạy thêm**.
Giữ `NUM_ENVS` giống run trước; file YAML ghi lại cấu hình đầy đủ.

Sao lưu trên máy cá nhân trước khi Destroy:

```bash
mkdir -p logs/vast-backup
rsync -rt --info=progress2 \
  -e 'ssh -i ~/.ssh/mjlab_vast -p PORT' \
  USER@HOST:/workspace/mjlab/logs/ ./logs/vast-backup/
```

Xác nhận checkpoint đã tải về, sau đó vào Vast **Instances → Destroy/Delete**.
Nếu dùng volume riêng, kiểm tra và xử lý volume không còn cần vì nó có thể
tiếp tục phát sinh phí.

## Kiểm chứng local ngày 20/09/2026

Máy local: AMD tích hợp, khoảng 7 GB RAM, không có NVIDIA/CUDA.
Dùng môi trường sẵn có `/home/TranPhuNghia_20233871/venvs/mjlab`,
đặt `PYTHONPATH` tới `src` của repo hiện tại. Không cài CUDA trên máy này.

- G1 flat velocity: **4 môi trường × 2 iteration = 192 transition**, thành công.
  Checkpoint `logs/rsl_rl/g1_velocity/2026-09-20_20-49-50_cpu-g1-smoke/model_1.pt`.
  Toàn bộ tensor checkpoint hữu hạn; ONNX xuất ra vượt qua `onnx.checker`.
  Đây chỉ là smoke test, chưa phải policy biết đi.
- Cartpole balance: **128 môi trường × 150 iteration = 614.400 transition**,
  vòng học khoảng 3 phút 9 giây, chưa gồm khởi tạo/biên dịch.
  Mean reward cuối 49,16; đây là reward training, không phải đánh giá độc lập.
  Checkpoint ở `logs/rsl_rl/cartpole/2026-09-20_20-46-08_cpu-pipeline-check/`.
- `tests/test_task_configs.py` và `tests/test_gpu_selection.py`: **19 passed**.
- Script Vast: `bash -n` đạt; trên máy không có NVIDIA, dừng đúng ở preflight.
- Chưa kiểm chứng CUDA, throughput, VRAM hoặc thời gian hội tụ G1 trên Vast.
  Chưa tạo instance, chưa thanh toán và chưa phát sinh phí thuê.

Console log local: `logs/training-setup/cpu-training.log` và
`logs/training-setup/g1-cpu-smoke.log`. Các thư mục `logs/` bị Git ignore;
cần sao lưu riêng nếu muốn giữ kết quả.
