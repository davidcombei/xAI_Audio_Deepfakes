"""Streamlit viewer for the simple-classifier experiment.

For one clip it shows the input spectrogram, the U-Net mask, 1 - mask,
mask * spectrogram and spectrogram * (1 - mask), and the mel logistic
regression's probability of "fake" for the original spectrogram,
mask * spectrogram and spectrogram * (1 - mask). Fake and bona fide clips are
on two separate pages.

The U-Net is the zero-fill one (the original ADDvisor masking), trained
against the mel logistic regression of the chosen setup.

Run from the repo root:

    CUDA_VISIBLE_DEVICES=0 python3 -m streamlit run simple_classifier/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: config.py, audio.py, new_masking/
sys.path.insert(0, str(Path(__file__).resolve().parent))         # this folder: common.py, mel_detector.py

import types

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

import config
from audio import AudioProcessor
from common import RUNS_DIR, complex_stft, detector_logits, detector_path, list_files, real_path
from mel_detector import MelLogRegDetector
from models import load_mask_predictor

# Streamlit's file watcher crashes on torch.classes; same fix as the repo's streamlit_app.py.
if isinstance(torch.classes, types.ModuleType):
    torch.classes.__path__ = []

# Final zero-fill U-Nets (see README.md).
UNET_CHECKPOINTS = {
    "freq-swapped": RUNS_DIR / "freq-swapped_zerofill" / "UNet_MP_epoch_60_loss_0.0066_bn_recalibrated.pth",
}

LAST_TIME_S = (config.N_TIME_FRAMES - 1) * config.HOP_LENGTH / config.SAMPLING_RATE
LAST_FREQUENCY_KHZ = (config.N_FREQ_BINS - 1) * (config.SAMPLING_RATE / 2) / config.N_FREQ_BINS / 1000
EXTENT = [0, LAST_TIME_S, 0, LAST_FREQUENCY_KHZ]  # imshow axes: time (s) x frequency (kHz)
DB_RANGE = 100  # spectrograms are shown from (loudest bin - 100 dB) to the loudest bin


@st.cache_resource(show_spinner="Loading models ...")
def load_models(setup_name):
    """Audio processor, mel logistic regression and zero-fill U-Net of one setup."""
    device = config.get_device()
    ap = AudioProcessor(device=device)
    detector = MelLogRegDetector(detector_path(setup_name)).to(device)
    unet = load_mask_predictor(UNET_CHECKPOINTS[setup_name], device)
    return ap, detector, unet


@st.cache_data
def fake_files(setup_name):
    """Repo-relative paths of every fake clip of the setup."""
    return list_files(config.get_setup(setup_name))


@st.cache_data
def real_files(setup_name):
    """Repo-relative paths of the bona fide clips the setup's fakes were made from."""
    paths = []
    seen = set()
    for fake_path in fake_files(setup_name):
        path = str(real_path(fake_path).relative_to(config.PROJECT_ROOT))
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def probability_fake(ap, detector, spectrogram):
    """Logistic regression probability of "fake" for one complex spectrogram (1, 512, 248)."""
    logit = detector_logits(ap, detector, spectrogram)  # (1,)
    return torch.sigmoid(logit).item()


@torch.no_grad()
def explain(ap, detector, unet, path):
    """Spectrograms, masks and probabilities of one clip."""
    waveform, _ = ap.load_audio(config.PROJECT_ROOT / path)  # (80000,)
    waveform = waveform.to(ap.device).unsqueeze(0)            # (1, 80000)
    spectrogram = complex_stft(ap, waveform)                  # (1, 512, 248) complex

    mask = unet(spectrogram.abs().unsqueeze(1)).squeeze(1)    # (1, 512, 248), values in [0, 1]
    kept = mask * spectrogram                                 # mask * spec
    removed = (1 - mask) * spectrogram                        # spec * (1 - mask)

    return {
        "spec": spectrogram[0].abs().cpu().numpy(),
        "mask": mask[0].cpu().numpy(),
        "kept": kept[0].abs().cpu().numpy(),
        "removed": removed[0].abs().cpu().numpy(),
        "p_original": probability_fake(ap, detector, spectrogram),
        "p_kept": probability_fake(ap, detector, kept),
        "p_removed": probability_fake(ap, detector, removed),
    }


def to_db(magnitude):
    """Magnitude -> decibels (the 1e-10 keeps log10 finite on zeroed bins)."""
    return 20 * np.log10(magnitude + 1e-10)


def plot_spectrogram(magnitude, loudest_db, title):
    fig, ax = plt.subplots(figsize=(5, 3.2), constrained_layout=True)
    image = ax.imshow(to_db(magnitude), origin="lower", aspect="auto", cmap="magma", extent=EXTENT,
                      vmin=loudest_db - DB_RANGE, vmax=loudest_db)
    fig.colorbar(image, ax=ax, label="dB")
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (kHz)")
    return fig


def plot_mask(mask, title):
    fig, ax = plt.subplots(figsize=(5, 3.2), constrained_layout=True)
    image = ax.imshow(mask, origin="lower", aspect="auto", cmap="viridis", extent=EXTENT, vmin=0, vmax=1)
    fig.colorbar(image, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (kHz)")
    return fig


def show_clip(paths):
    """Clip picker, the five images and the three probabilities."""
    setup_name = st.session_state["setup_name"]
    ap, detector, unet = load_models(setup_name)

    index = st.number_input(f"Clip index (0 - {len(paths) - 1})", min_value=0, max_value=len(paths) - 1, value=0)
    path = paths[index]
    st.subheader(path)
    out = explain(ap, detector, unet, path)
    loudest_db = to_db(out["spec"]).max()  # same colour scale for the three spectrograms

    col1, col2, col3 = st.columns(3)
    col1.pyplot(plot_spectrogram(out["spec"], loudest_db, "Input spectrogram"))
    col2.pyplot(plot_mask(out["mask"], "Mask"))
    col3.pyplot(plot_mask(1 - out["mask"], "1 - Mask"))

    col1, col2, col3 = st.columns(3)
    col1.pyplot(plot_spectrogram(out["kept"], loudest_db, "Mask * Spec"))
    col2.pyplot(plot_spectrogram(out["removed"], loudest_db, "Spec * (1 - Mask)"))

    st.markdown("**LogReg probability of fake**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Original spectrogram", f"{out['p_original']:.3f}")
    col2.metric("Mask * Spec", f"{out['p_kept']:.3f}")
    col3.metric("Spec * (1 - Mask)", f"{out['p_removed']:.3f}")
    plt.close("all")


def fake_page():
    st.title("Fake spectrograms")
    show_clip(fake_files(st.session_state["setup_name"]))


def real_page():
    st.title("Real spectrograms")
    show_clip(real_files(st.session_state["setup_name"]))


st.set_page_config(layout="wide", page_title="Simple classifier")
st.session_state["setup_name"] = st.sidebar.selectbox("Setup", ["freq-swapped"])
page = st.navigation([st.Page(fake_page, title="Fake", url_path="fake"),
                      st.Page(real_page, title="Real", url_path="real")])
page.run()
