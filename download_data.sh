#!/usr/bin/env bash
# Download checkpoints + datasets for ADDvisor from the Hugging Face Hub.
# Requires: wget, tar, zstd  (sudo apt-get install -y wget tar zstd)
# Run from the repo root:  bash download_data.sh
set -euo pipefail

BASE="https://huggingface.co/datasets/DavidCombei/xAI_Audio_Deepfakes/resolve/main"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "==> Checkpoints (config.py)"
mkdir -p ckpts/ckpts_realisticDeepfakes ckpts/ckpts_freqSwapDeepfakes ckpts/logReg_ckpts
wget -q --show-progress -O ckpts/ckpts_realisticDeepfakes/UNet_MP_epoch_41_loss_0.3316.pth \
  "$BASE/ckpts/ckpts_realisticDeepfakes/UNet_MP_epoch_41_loss_0.3316.pth"
wget -q --show-progress -O ckpts/ckpts_freqSwapDeepfakes/UNet_MP_epoch_74_loss_0.0182.pth \
  "$BASE/ckpts/ckpts_freqSwapDeepfakes/UNet_MP_epoch_74_loss_0.0182.pth"
for f in logReg_MLAAD_M-AILabs logReg_vocoded_anyband_controlled; do
  wget -q --show-progress -O "ckpts/logReg_ckpts/$f.joblib" "$BASE/ckpts/logReg_ckpts/$f.joblib"
done

echo "==> wav2vec2 feature extractor"
wget -q --show-progress -O wav2vec2.tar.zst "$BASE/models/wav2vec2-xls-r-2b_truncated.tar.zst"
tar -I zstd -xf wav2vec2.tar.zst && rm wav2vec2.tar.zst   # -> models/wav2vec2-xls-r-2b_truncated/

echo "==> Datasets (LJSpeech_hifigan is the controlled-setup training data)"
for d in LJSpeech_freq_swap mlaad_subset LJSpeech_hifigan; do
  echo "    $d ..."
  wget -q --show-progress -O "$d.tar.zst" "$BASE/data/$d.tar.zst"
  tar -I zstd -xf "$d.tar.zst" && rm "$d.tar.zst"
done

echo "==> Done. Folders are in place for config.py."
