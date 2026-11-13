#!/usr/bin/env bash
# Bootstrap the optimized fill of one setup: train the U-Net with the current fill, fine-tune the
# fill on the frozen U-Net's masks, and repeat.
#
#   ./bootstrap_fill.sh freq-swapped-single 0            setup, GPU
#   ROUNDS=4 EPOCHS_PER_ROUND=2 FILL_STEPS=800 ./bootstrap_fill.sh realistic 4
#
# Each round:
#   1. train_mask.py --fill optimized for EPOCHS_PER_ROUND more epochs (runs/<setup>_optimizedfill_bootstrap)
#   2. learn_neutral_fill.py --init <current fill> --mask-ckpt <that U-Net> for FILL_STEPS steps
# output/neutral_fill/neutral_fill_<setup>.* is always the latest fill; the fill each round started
# from is kept as neutral_fill_<setup>_round<N>.* (round0 = before any fine-tuning).
# The U-Net trains on FILELIST (default output/neutral_fill/bootstrap/<setup>_train.txt), which
# leaves out the clips learn_neutral_fill.py holds out.

set -eu
cd "$(dirname "$0")"

SETUP=$1
GPU=$2
ROUNDS=${ROUNDS:-4}
EPOCHS_PER_ROUND=${EPOCHS_PER_ROUND:-2}
FILL_STEPS=${FILL_STEPS:-800}
FILL_LR=${FILL_LR:-3e-3}
BATCH_SIZE=${BATCH_SIZE:-15}
WORKERS=${WORKERS:-4}
PYTHON=${PYTHON:-.venv/bin/python}
FILELIST=${FILELIST:-output/neutral_fill/bootstrap/${SETUP}_train.txt}
RUN=runs/${SETUP}_optimizedfill_bootstrap
FILL=output/neutral_fill/neutral_fill_${SETUP}

for round in $(seq 1 "$ROUNDS"); do
    epochs=$((round * EPOCHS_PER_ROUND))
    echo "[$(date +%H:%M)] $SETUP round $round: U-Net to epoch $epochs"
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" train_mask.py --setup "$SETUP" --fill optimized \
        --filelist "$FILELIST" --epochs "$epochs" --batch-size "$BATCH_SIZE" --workers "$WORKERS" \
        --out-dir "$RUN" --resume
    ckpt=$(ls "$RUN"/UNet_MP_epoch_"${epochs}"_loss_*.pth | head -1)

    echo "[$(date +%H:%M)] $SETUP round $round: fill on the masks of $ckpt"
    for ext in wav npy txt; do
        cp "$FILL.$ext" "${FILL}_round$((round - 1)).$ext"
    done
    CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" learn_neutral_fill.py --setups "$SETUP" --name "neutral_fill_${SETUP}" \
        --init "${FILL}_round$((round - 1)).wav" --mask-ckpt "$ckpt" --steps "$FILL_STEPS" --lr "$FILL_LR" \
        --pairs-per-setup 12 --n-train 768
done
echo "[$(date +%H:%M)] $SETUP done"
