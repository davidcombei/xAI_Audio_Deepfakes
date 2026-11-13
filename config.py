from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent

CKPTS_DIR = PROJECT_ROOT / "ckpts"
CKPTS_DIR_RUNS = PROJECT_ROOT / "runs"
METADATA_DIR = PROJECT_ROOT / "metadata"
FEATURES_DIR = PROJECT_ROOT / "features"
NPY_DIR = PROJECT_ROOT / "npy"
VIZ_DIR = PROJECT_ROOT / "visualizations"
RUNS_DIR = PROJECT_ROOT / "runs"  


TRAIN_AUDIO_DIR = PROJECT_ROOT / "LJSpeech_vocoded22K"  
VOCODED_DIR = PROJECT_ROOT / "LJSpeech_hifigan"        


W2V2_LOCAL_DIR = PROJECT_ROOT / "models" / "wav2vec2-xls-r-2b_truncated"
W2V2_HF_ID = "facebook/wav2vec2-xls-r-2b"
W2V2_HIDDEN_LAYER = 9   
FEATURE_DIM = 1920      




SAMPLING_RATE = 16_000
N_MELS = 80
AUDIO_LENGTH = 5  # seconds

# STFT used to make the freq swaps (vocoder.py). The mask pipeline uses the same
# STFT (window covers the whole FFT, win == n_fft; torch's default rectangular
# window, as in vocoder.py). The shipped freq swaps are 22.05 kHz files, so after
# resampling to 16 kHz the bands line up closely, not exactly, with the bins
# (98-99 % of the fake-real difference falls inside the named band).
SWAP_N_FFT = 1024
SWAP_HOP = 256
SWAP_WIN = 1024

N_FFT = SWAP_N_FFT
HOP_LENGTH = SWAP_HOP
WIN_LENGTH = SWAP_WIN

# Full spectrogram, nothing cropped: 513 x 313 for a 5 s clip at 16 kHz.
N_FREQ_BINS = N_FFT // 2 + 1                                        # 513
N_TIME_FRAMES = 1 + (SAMPLING_RATE * AUDIO_LENGTH) // HOP_LENGTH    # 313



@dataclass(frozen=True)
class Setup:


    name: str
    data_dir: Path
    filelist: Path
    mask_ckpt: Path | None
    detector_ckpt: Path
    localization: str | None = None
    # Use the ground-truth masks of `localization` instead of the U-Net.
    ground_truth_masks: bool = False
    # Bona fide clips the fakes were made from (see data.real_path).
    # Needed for real fill; None = no bona fide reference, only zero fill works.
    real_dir: Path | None = None
    # Which wavs in data_dir belong to this setup (used by train_mask.py).
    file_pattern: str = "*.wav"


# NOTE: the mask_ckpt U-Nets below were trained on the old 512 x 248 STFT
# (n_fft 1024, hop 322, win 644) with the old decoder. They do not load into the
# current UNet; retrain with train_mask.py and point mask_ckpt to the new files.
SETUPS: dict[str, Setup] = {
    "realistic": Setup(
        name="realistic",
        data_dir=PROJECT_ROOT / "mlaad_subset",
        # The 27171 files the U-Net in runs/realistic_realfill was trained on
        # (every wav in mlaad_subset/, each one has an M-AILABS clip in mailabs_real/).
        filelist=METADATA_DIR / "realistic-all.txt",
        mask_ckpt=CKPTS_DIR_RUNS / "realistic_avgfill" / "UNet_MP_epoch_100_loss_0.1180.pth",
        detector_ckpt=CKPTS_DIR / "logReg_ckpts" / "logReg_MLAAD_M-AILabs.joblib",
        localization=None,
        # Links to the M-AILABS clips, made by link_mailabs_reals.py. These clips are
        # not time-aligned with the fakes (see masking.py).
        real_dir=PROJECT_ROOT / "mailabs_real",
    ),
    "freq-swapped": Setup(
        name="freq-swapped",
        data_dir=PROJECT_ROOT / "LJSpeech_freq_swap",
        filelist=METADATA_DIR / "freq-swap-all.txt",  # all 40000 swaps (8 bands x 5000 clips)
       # mask_ckpt=CKPTS_DIR / "ckpts_freqSwapDeepfakes" / "UNet_MP_epoch_74_loss_0.0182.pth",
        mask_ckpt = CKPTS_DIR_RUNS / "freq-swapped_avgfill" / "UNet_MP_epoch_100_loss_0.0508.pth",
        detector_ckpt=CKPTS_DIR / "logReg_ckpts" / "logReg_vocoded_anyband_controlled.joblib",
        localization="freq",
        real_dir=TRAIN_AUDIO_DIR,
    ),
    # Only the clips whose 1-2 kHz band was swapped, with a detector trained on
    # that band alone.
    "freq-swapped-single": Setup(
        name="freq-swapped-single",
        data_dir=PROJECT_ROOT / "LJSpeech_freq_swap",
        filelist=METADATA_DIR / "freq-swap-single-all.txt",  # all 5000 1-2 kHz swaps
        mask_ckpt=CKPTS_DIR_RUNS / "freq-swapped-single_avgfill" / "UNet_MP_epoch_100_loss_0.1480.pth",  
        detector_ckpt=CKPTS_DIR / "logReg_ckpts" / "logReg_vocoded_1-2kHz_controlled.joblib",
        localization="freq",
        real_dir=TRAIN_AUDIO_DIR,
        file_pattern="*_vocoded_1000-2000.wav",
    ),
}

# Oracle baselines: the swap setups with the U-Net replaced by the known
# manipulated region, i.e. what a perfect mask predictor would score.
SETUPS["ground_truth_frequency"] = replace(
    SETUPS["freq-swapped"], name="ground_truth_frequency", mask_ckpt=None, ground_truth_masks=True
)

SETUP_NAMES = tuple(SETUPS)


def get_setup(name: str) -> Setup:
    try:
        return SETUPS[name]
    except KeyError:
        raise ValueError(
            f"unknown setup {name!r}. Choose one of: {', '.join(SETUP_NAMES)}"
        ) from None


def setup_key(setup) -> str:
    """Short key used in output filenames: 'realistic' / 'freq' / 'freq-single'."""
    name = setup.name if isinstance(setup, Setup) else setup
    key = name.split("-")[0]
    if name.endswith("-single"):
        key = key + "-single"
    return key


def masked_features_dir(setup) -> Path:
    return FEATURES_DIR / f"masked_features_{setup_key(setup)}"


def get_device(prefer: str | None = None) -> torch.device:
    if prefer is not None:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
