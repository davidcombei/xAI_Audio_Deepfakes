#!/usr/bin/env bash
# The whole simple-classifier experiment, run from the repo root.
# --gpu_ids is needed: the machine's default accelerate config pins its own GPU list.
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. train the mel logistic regression detectors (a few minutes each)
CUDA_VISIBLE_DEVICES=0 python3 simple_classifier/train_detector.py --setup freq-swapped
CUDA_VISIBLE_DEVICES=0 python3 simple_classifier/sanity_check.py --n 200

# 2. train the U-Net against them, zero fill and real fill (2 runs at once, 2 GPUs each)
accelerate launch --multi_gpu --gpu_ids 0,1 --num_processes 2 --main_process_port 29521 \
    simple_classifier/train_mask.py --setup freq-swapped --fill zero --epochs 60 --resume &
accelerate launch --multi_gpu --gpu_ids 2,3 --num_processes 2 --main_process_port 29522 \
    simple_classifier/train_mask.py --setup freq-swapped --fill real --epochs 60 --resume &
wait

# 3. recompute the BatchNorm statistics of the final checkpoints (weights unchanged; see new_masking/README.md)
for run in freq-swapped_zerofill:60 freq-swapped_realfill:60; do
  name=${run%%:*}; epoch=${run##*:}; setup=${name%_*}
  CUDA_VISIBLE_DEVICES=0 python3 new_masking/recalibrate_bn.py --setup "$setup" \
      --ckpt "$(ls simple_classifier/runs/$name/UNet_MP_epoch_${epoch}_loss_*[0-9].pth)"
done
ckpt() { ls simple_classifier/runs/$1/UNet_MP_epoch_$2_*_bn_recalibrated.pth; }

# 4. evaluate on all files, one shard per GPU, then merge the shards
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python3 simple_classifier/evaluate.py --setup freq-swapped \
      --zerofill-ckpt "$(ckpt freq-swapped_zerofill 60)" --realfill-ckpt "$(ckpt freq-swapped_realfill 60)" \
      --shard $i --num-shards 8 &
done; wait
python3 simple_classifier/evaluate.py --setup freq-swapped --summarize-only

# 5. figures
CUDA_VISIBLE_DEVICES=0 python3 simple_classifier/figures.py \
    --freq-zerofill-ckpt "$(ckpt freq-swapped_zerofill 60)" --freq-realfill-ckpt "$(ckpt freq-swapped_realfill 60)"
