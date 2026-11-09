#!/usr/bin/env bash
# The whole real-fill experiment, run from the repo root.
# --gpu_ids is needed: the machine's default accelerate config pins its own GPU list.
set -euo pipefail
cd "$(dirname "$0")/.."

# 0. checks before training
CUDA_VISIBLE_DEVICES=0 python3 new_masking/sanity_check.py --n 60

# 1. retrain the U-Net with real fill (both setups at once: 5 + 3 GPUs)
accelerate launch --multi_gpu --gpu_ids 0,1,2,3,4 --num_processes 5 --main_process_port 29511 \
    new_masking/train_mask.py --setup freq-swapped --epochs 40 --batch-size 15 --resume &
accelerate launch --multi_gpu --gpu_ids 5,6,7 --num_processes 3 --main_process_port 29512 \
    new_masking/train_mask.py --setup time-swapped --epochs 150 --batch-size 15 --resume &
wait

# 2. recompute the BatchNorm statistics of the final checkpoints (weights unchanged)
CUDA_VISIBLE_DEVICES=0 python3 new_masking/recalibrate_bn.py --setup freq-swapped \
    --ckpt "$(ls new_masking/runs/freq-swapped_realfill/UNet_MP_epoch_40_loss_*[0-9].pth)"
CUDA_VISIBLE_DEVICES=0 python3 new_masking/recalibrate_bn.py --setup time-swapped \
    --ckpt "$(ls new_masking/runs/time-swapped_realfill/UNet_MP_epoch_150_loss_*[0-9].pth)"
FREQ_CKPT=$(ls new_masking/runs/freq-swapped_realfill/UNet_MP_epoch_40_*_bn_recalibrated.pth)
TIME_CKPT=$(ls new_masking/runs/time-swapped_realfill/UNet_MP_epoch_150_*_bn_recalibrated.pth)

# 3. evaluate on all files, one shard per GPU, then merge the shards
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python3 new_masking/evaluate.py --setup freq-swapped \
      --realfill-ckpt "$FREQ_CKPT" --shard $i --num-shards 8 &
done; wait
python3 new_masking/evaluate.py --setup freq-swapped --summarize-only

for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python3 new_masking/evaluate.py --setup time-swapped \
      --realfill-ckpt "$TIME_CKPT" --shard $i --num-shards 8 &
done; wait
python3 new_masking/evaluate.py --setup time-swapped --summarize-only

# 4. figures
CUDA_VISIBLE_DEVICES=0 python3 new_masking/figures.py --freq-ckpt "$FREQ_CKPT" --time-ckpt "$TIME_CKPT"
