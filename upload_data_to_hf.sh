#!/usr/bin/env bash
# Upload checkpoints + datasets used by config.py to a Hugging Face dataset repo.
# Run AFTER `hf auth login` (you are already logged in as DavidCombei).
set -euo pipefail

ROOT="/mnt/QNAP/comdav/xAI_deepfakes/addvisor"
REPO="DavidCombei/xAI_Audio_Deepfakes"      # dataset repo
STAGE="$ROOT/hf_stage"
cd "$ROOT"
mkdir -p "$STAGE"

echo "### Creating dataset repo $REPO (ok if it already exists)"
hf repo create "$REPO" --repo-type dataset --exist-ok

upload_file () {  # <local> <path_in_repo>
  echo "### upload file: $2"
  hf upload "$REPO" "$1" "$2" --repo-type dataset
}

tar_and_upload () {  # <dir relative to ROOT> <path_in_repo without extension>
  local dir="$1" dest="$2"
  local tarball="$STAGE/$(basename "$dir").tar.zst"
  echo "### tar  $dir  ->  $tarball"
  tar -C "$ROOT" -cf - "$dir" | zstd -T0 -3 -q -o "$tarball" -f
  echo "### upload archive: ${dest}.tar.zst  ($(du -h "$tarball" | cut -f1))"
  hf upload "$REPO" "$tarball" "${dest}.tar.zst" --repo-type dataset
  rm -f "$tarball"
}

echo "==================== CHECKPOINTS (config.py) ===================="
upload_file ckpts/ckpts_realisticDeepfakes/UNet_MP_epoch_41_loss_0.3316.pth   ckpts/ckpts_realisticDeepfakes/UNet_MP_epoch_41_loss_0.3316.pth
upload_file ckpts/ckpts_freqSwapDeepfakes/UNet_MP_epoch_74_loss_0.0182.pth    ckpts/ckpts_freqSwapDeepfakes/UNet_MP_epoch_74_loss_0.0182.pth
upload_file ckpts/ckpts_timeSwapDeepfakes_CORRECTED/UNet_MP_epoch_173_loss_0.3314.pth ckpts/ckpts_timeSwapDeepfakes_CORRECTED/UNet_MP_epoch_173_loss_0.3314.pth
upload_file ckpts/logReg_ckpts/logReg_MLAAD_M-AILabs.joblib            ckpts/logReg_ckpts/logReg_MLAAD_M-AILabs.joblib
upload_file ckpts/logReg_ckpts/logReg_vocoded_anyband_controlled.joblib ckpts/logReg_ckpts/logReg_vocoded_anyband_controlled.joblib
upload_file ckpts/logReg_ckpts/logReg_vocoded_TIMESWAP_CORRECTED.joblib ckpts/logReg_ckpts/logReg_vocoded_TIMESWAP_CORRECTED.joblib

echo "==================== WAV2VEC2 MODEL ===================="
tar_and_upload models/wav2vec2-xls-r-2b_truncated models/wav2vec2-xls-r-2b_truncated

echo "==================== DATASETS (config.py setups) ===================="
tar_and_upload LJSpeech_freq_swap data/LJSpeech_freq_swap
tar_and_upload LJSpeech_time_swap data/LJSpeech_time_swap
tar_and_upload mlaad_subset       data/mlaad_subset

echo "==================== TRAINING DATA (controlled setup) ===================="
tar_and_upload LJSpeech_hifigan   data/LJSpeech_hifigan

rmdir "$STAGE" 2>/dev/null || true
echo "### DONE -> https://huggingface.co/datasets/$REPO"
