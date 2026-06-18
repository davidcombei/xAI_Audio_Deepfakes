from __future__ import annotations

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

import config
from audio import AudioProcessor
from data import read_lines
from metrics import GROUND_TRUTH

st.set_page_config(layout="wide")
sns.set(style="white", context="poster", font="Arial")

SR = config.SAMPLING_RATE
HOP_LENGTH = config.HOP_LENGTH
EXPL_DIR = config.PROJECT_ROOT / "output" / "explanations"
OUT_DIR = config.PROJECT_ROOT / "output" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ap = AudioProcessor()  


def load_explanation(setup, method, filename):
    path = EXPL_DIR / setup / method / (os.path.basename(filename) + ".npy")
    mask = np.abs(np.load(path))
    return np.sqrt(mask / (mask.max() + 1e-8))


def load_groundtruth(setup, filename):
    kind = config.get_setup(setup).localization
    shape = (config.N_FREQ_BINS, config.N_TIME_FRAMES)
    return GROUND_TRUTH[kind](filename, shape).numpy()


def load_spectrogram(filename):
    audio, _ = ap.load_audio(config.PROJECT_ROOT / filename)
    _, magnitude, _ = ap.compute_stft(audio, has_batch_dim=False)
    return magnitude.cpu().numpy()


def load_matrix(setup, method, filename):
    if method == "input":
        return load_spectrogram(filename)
    if method == "groundtruth":
        return load_groundtruth(setup, filename)
    return load_explanation(setup, method, filename)


def make_image(setup, method, filename):
    mask = load_matrix(setup, method, filename)
    fig, ax = plt.subplots(figsize=(10, 6))
    duration = (mask.shape[1] * HOP_LENGTH) / SR
    ax.imshow(mask, aspect="auto", origin="lower", extent=[0, duration, 0, SR / 2],
              vmin=0, vmax=1, cmap="viridis")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    name = os.path.splitext(os.path.basename(filename))[0]
    fig.savefig(OUT_DIR / f"{setup}-{method}-{name}.png", bbox_inches="tight")
    return fig


def figure(setup, methods, filelist, k=2, contains=None):
    st.header(setup)
    files = read_lines(config.METADATA_DIR / filelist)
    if contains:
        files = [f for f in files if contains in f]
    for filename in random.sample(files, min(k, len(files))):
        cols = st.columns(len(methods))
        for col, method in zip(cols, methods):
            with col:
                st.caption(method)
                try:
                    st.pyplot(make_image(setup, method, filename))
                except FileNotFoundError:
                    st.warning(f"missing cache for {method}; run evaluate.py first")


seed = st.selectbox("Random seed", [0, 1, 2, 3, 4], index=1)
random.seed(seed)

figure("realistic", ["input", "saliency", "input-x-gradient", "gradient-shap", "optimized"],
       "realistic-subset.txt", k=2)
figure("freq-swapped", ["input", "groundtruth", "optimized"],
       "freq-swap-subset.txt", k=1, contains="2000-3000")
figure("time-swapped", ["input", "groundtruth", "optimized"],
       "time-swap-subset.txt", k=1)
