"""Streamlit viewer for the real-fill masking experiment.

For one clip it shows the input spectrogram, the U-Net mask, 1 - mask, the two
real-fill inputs and the bona fide reference clip, and the detector's
(wav2vec2 + logistic regression) probability of "fake" for each of them:

    keep   = mask * X_fake + (1 - mask) * X_real
    remove = (1 - mask) * X_fake + mask * X_real

Fake and bona fide clips are on two separate pages. A bona fide clip is its
own reference, so with real fill both inputs equal the clip itself; the Real
page only shows the mask.

The U-Net is the real-fill one in ckpts/, epoch 11 (UNET_CHECKPOINT below).

Run from the repo root:

    CUDA_VISIBLE_DEVICES=0 python3 -m streamlit run new_masking/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: config.py, audio.py, models.py
sys.path.insert(0, str(Path(__file__).resolve().parent))         # this folder: common.py

import types

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

import config
from audio import AudioProcessor
from common import (
    EXPERIMENT_DIR, complex_stft, detector_logits, keep_masked_part, keep_only_needed_wav2vec2_layers,
    list_files, real_path, remove_masked_part,
)
from models import TorchLogReg, load_mask_predictor

# Streamlit's file watcher crashes on torch.classes; same fix as the repo's streamlit_app.py.
if isinstance(torch.classes, types.ModuleType):
    torch.classes.__path__ = []

# Real-fill U-Net, epoch 11. Removing its mask flips the freq-swapped detector
# (p_fake 1.00 -> 0.00 on 12 clips) but only partly the time-swapped one (0.98 -> 0.15).
UNET_CHECKPOINT = EXPERIMENT_DIR / "ckpts" / "UNet_MP_epoch_11_loss_0.1363.pth"

LAST_TIME_S = (config.N_TIME_FRAMES - 1) * config.HOP_LENGTH / config.SAMPLING_RATE
LAST_FREQUENCY_KHZ = (config.N_FREQ_BINS - 1) * (config.SAMPLING_RATE / 2) / config.N_FREQ_BINS / 1000
EXTENT = [0, LAST_TIME_S, 0, LAST_FREQUENCY_KHZ]  # imshow axes: time (s) x frequency (kHz)
DB_RANGE = 100  # spectrograms are shown from (loudest bin - 100 dB) to the loudest bin


@st.cache_resource(show_spinner="Loading U-Net ...")
def load_unet():
    """The real-fill U-Net (the same for every setup)."""
    return load_mask_predictor(UNET_CHECKPOINT, config.get_device())


@st.cache_resource(show_spinner="Loading wav2vec2 ...")
def load_audio_processor():
    ap = AudioProcessor(device=config.get_device())
    keep_only_needed_wav2vec2_layers(ap)  # same features, ~4x faster
    return ap


@st.cache_resource(show_spinner="Loading detector ...")
def load_detector(setup_name):
    """Logistic regression on wav2vec2 features of one setup."""
    return TorchLogReg(config.get_setup(setup_name).detector_ckpt).to(config.get_device())


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
    """Detector probability of "fake" for one complex spectrogram (1, 512, 248)."""
    logit = detector_logits(ap, detector, spectrogram)  # (1,)
    return torch.sigmoid(logit).item()


def load_spectrogram(ap, path):
    """Complex STFT (1, 512, 248) of one clip."""
    waveform, _ = ap.load_audio(path)            # (80000,)
    waveform = waveform.to(ap.device).unsqueeze(0)  # (1, 80000)
    return complex_stft(ap, waveform)


@torch.no_grad()
def explain(ap, detector, unet, path, is_fake):
    """Spectrograms, masks and probabilities of one clip.

    For a fake clip the bona fide clip it was made from fills the masked-out
    bins. A bona fide clip has no other reference, so only its mask is computed.
    """
    spectrogram = load_spectrogram(ap, config.PROJECT_ROOT / path)  # (1, 512, 248) complex
    mask = unet(spectrogram.abs().unsqueeze(1)).squeeze(1)           # (1, 512, 248), values in [0, 1]
    out = {
        "spec": spectrogram[0].abs().cpu().numpy(),
        "mask": mask[0].cpu().numpy(),
        "p_original": probability_fake(ap, detector, spectrogram),
    }
    if not is_fake:
        return out

    reference = load_spectrogram(ap, real_path(path))                     # (1, 512, 248) complex
    kept = keep_masked_part(mask, spectrogram, reference, "real")         # mask * fake + (1 - mask) * real
    removed = remove_masked_part(mask, spectrogram, reference, "real")    # (1 - mask) * fake + mask * real
    out.update({
        "reference": reference[0].abs().cpu().numpy(),
        "kept": kept[0].abs().cpu().numpy(),
        "removed": removed[0].abs().cpu().numpy(),
        "p_reference": probability_fake(ap, detector, reference),
        "p_kept": probability_fake(ap, detector, kept),
        "p_removed": probability_fake(ap, detector, removed),
    })
    return out


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


def show_clip(paths, is_fake):
    """Clip picker, the images and the probabilities."""
    setup_name = st.session_state["setup_name"]
    ap = load_audio_processor()
    detector = load_detector(setup_name)
    unet = load_unet()

    index = st.number_input(f"Clip index (0 - {len(paths) - 1})", min_value=0, max_value=len(paths) - 1, value=0)
    path = paths[index]
    st.subheader(path)
    out = explain(ap, detector, unet, path, is_fake)
    loudest_db = to_db(out["spec"]).max()  # same colour scale for every spectrogram

    col1, col2, col3 = st.columns(3)
    col1.pyplot(plot_spectrogram(out["spec"], loudest_db, "Input spectrogram"))
    col2.pyplot(plot_mask(out["mask"], "Mask"))
    col3.pyplot(plot_mask(1 - out["mask"], "1 - Mask"))

    if not is_fake:
        st.caption("A bona fide clip is its own reference: with real fill, Mask * Fake + (1 - Mask) * Real "
                   "and (1 - Mask) * Fake + Mask * Real both equal the input.")
        st.markdown("**Detector probability of fake**")
        st.metric("Original spectrogram", f"{out['p_original']:.3f}")
        plt.close("all")
        return

    st.caption(f"Bona fide reference: {real_path(path).relative_to(config.PROJECT_ROOT)}")
    col1, col2, col3 = st.columns(3)
    col1.pyplot(plot_spectrogram(out["kept"], loudest_db, "Mask * Fake + (1 - Mask) * Real"))
    col2.pyplot(plot_spectrogram(out["removed"], loudest_db, "(1 - Mask) * Fake + Mask * Real"))
    col3.pyplot(plot_spectrogram(out["reference"], loudest_db, "Bona fide reference"))

    st.markdown("**Detector probability of fake**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Original spectrogram", f"{out['p_original']:.3f}")
    col2.metric("Mask * Fake + (1 - Mask) * Real", f"{out['p_kept']:.3f}")
    col3.metric("(1 - Mask) * Fake + Mask * Real", f"{out['p_removed']:.3f}")
    col4.metric("Bona fide reference", f"{out['p_reference']:.3f}")
    plt.close("all")


def fake_page():
    st.title("Fake spectrograms")
    show_clip(fake_files(st.session_state["setup_name"]), is_fake=True)


def real_page():
    st.title("Real spectrograms")
    show_clip(real_files(st.session_state["setup_name"]), is_fake=False)


st.set_page_config(layout="wide", page_title="Real-fill masking")
st.session_state["setup_name"] = st.sidebar.selectbox("Setup", ["freq-swapped", "time-swapped"])
st.sidebar.markdown(f"**U-Net (real fill)**  \n`{UNET_CHECKPOINT.relative_to(config.PROJECT_ROOT)}`")
if st.session_state["setup_name"] == "time-swapped":
    st.sidebar.warning("This U-Net looks trained on freq-swapped: on time-swapped clips "
                       "removing its mask only partly flips the detector.")
page = st.navigation([st.Page(fake_page, title="Fake", url_path="fake"),
                      st.Page(real_page, title="Real", url_path="real")])
page.run()
