from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent

CKPTS_DIR = PROJECT_ROOT / "ckpts"
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


DEFAULT_DETECTOR_CKPT = CKPTS_DIR / "logReg_ckpts" / "logReg_vocoded_TIMESWAP_CORRECTED.joblib"


SAMPLING_RATE = 16_000
N_FFT = 1024
HOP_LENGTH = 322
WIN_LENGTH = 644
N_MELS = 80
AUDIO_LENGTH = 5  #


N_FREQ_BINS = N_FFT // 2         
N_TIME_FRAMES = 248



@dataclass(frozen=True)
class Setup:


    name: str
    data_dir: Path
    filelist: Path
    mask_ckpt: Path
    detector_ckpt: Path
    localization: str | None = None


SETUPS: dict[str, Setup] = {
    "realistic": Setup(
        name="realistic",
        data_dir=PROJECT_ROOT / "mlaad_subset",
        filelist=METADATA_DIR / "realistic-subset.txt",
        mask_ckpt=CKPTS_DIR / "ckpts_realisticDeepfakes" / "UNet_MP_epoch_41_loss_0.3316.pth",
        detector_ckpt=CKPTS_DIR / "logReg_ckpts" / "logReg_MLAAD_M-AILabs.joblib",
        localization=None,
    ),
    "freq-swapped": Setup(
        name="freq-swapped",
        data_dir=PROJECT_ROOT / "LJSpeech_freq_swap",
        filelist=METADATA_DIR / "freq-swap-subset.txt",
        mask_ckpt=CKPTS_DIR / "ckpts_freqSwapDeepfakes" / "UNet_MP_epoch_74_loss_0.0182.pth",
        detector_ckpt=CKPTS_DIR / "logReg_ckpts" / "logReg_vocoded_anyband_controlled.joblib",
        localization="freq",
    ),
    "time-swapped": Setup(
        name="time-swapped",
        data_dir=PROJECT_ROOT / "LJSpeech_time_swap",
        filelist=METADATA_DIR / "time-swap-subset.txt",
        mask_ckpt=CKPTS_DIR / "ckpts_timeSwapDeepfakes_CORRECTED" / "UNet_MP_epoch_173_loss_0.3314.pth",
        detector_ckpt=CKPTS_DIR / "logReg_ckpts" / "logReg_vocoded_TIMESWAP_CORRECTED.joblib",
        localization="time",
    ),
}

SETUP_NAMES = tuple(SETUPS)


def get_setup(name: str) -> Setup:
    try:
        return SETUPS[name]
    except KeyError:
        raise ValueError(
            f"unknown setup {name!r}. Choose one of: {', '.join(SETUP_NAMES)}"
        ) from None


def setup_key(setup) -> str:
    """Short key used in output filenames: 'realistic' / 'freq' / 'time'."""
    name = setup.name if isinstance(setup, Setup) else setup
    return name.split("-")[0]


def masked_features_dir(setup) -> Path:
    return FEATURES_DIR / f"masked_features_{setup_key(setup)}"


def get_device(prefer: str | None = None) -> torch.device:
    if prefer is not None:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
