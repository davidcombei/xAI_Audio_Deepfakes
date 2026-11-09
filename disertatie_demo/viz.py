from __future__ import annotations

import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import config


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _finish(fig, save_path, dpi=150):
    if save_path is None:
        return fig
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return None


def fig_to_png(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def plot_mask(
    mask,
    title: str = "",
    sr: int = config.SAMPLING_RATE,
    hop_length: int = config.HOP_LENGTH,
    save_path: str | os.PathLike | None = None,
):
    mask = _to_numpy(mask)
    duration = (mask.shape[1] * hop_length) / sr
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        mask, aspect="auto", origin="lower",
        extent=[0, duration, 0, sr / 2], vmin=0, vmax=1, cmap="viridis",
    )
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("freq (Hz)")
    ax.set_xlabel("time (s)")
    fig.colorbar(im, ax=ax, label="Mask value")
    return _finish(fig, save_path, dpi=100)


def plot_spectrogram(
    spec,
    title: str = "",
    sr: int = config.SAMPLING_RATE,
    hop_length: int = config.HOP_LENGTH,
    cmap: str = "viridis",
    log: bool = False,
    save_path: str | os.PathLike | None = None,
):
    spec = _to_numpy(spec)
    if log:
        spec = np.log1p(spec)
    n_freq_bins, num_frames = spec.shape

    freqs = np.linspace(0, sr / 2, n_freq_bins)
    desired_freqs = np.arange(0, sr // 2 + 1, 1000)
    yticks = [int(np.argmin(np.abs(freqs - f))) for f in desired_freqs]

    duration = (num_frames * hop_length) / sr
    desired_times = np.arange(0, duration + 0.5, 0.5)
    xticks = [int(t * sr / hop_length) for t in desired_times]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(spec, aspect="auto", origin="lower", cmap=cmap)
    ax.set_yticks(yticks, desired_freqs)
    ax.set_xticks(xticks, [f"{t:.1f}" for t in desired_times])
    ax.set_title(title)
    ax.set_ylabel("freq (Hz)")
    ax.set_xlabel("time (s)")
    fig.colorbar(im, ax=ax)
    return _finish(fig, save_path)


def plot_features(features, title: str = "", save_path: str | os.PathLike | None = None):
    features = _to_numpy(features)
    if features.ndim == 3:
        features = features[0]
    data = features.T
    f_min, f_max = float(data.min()), float(data.max())

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        data, aspect="auto", origin="lower", cmap="viridis_r", vmin=f_min, vmax=f_max
    )
    ax.set_title(f"{title} (min={f_min:.2f}, max={f_max:.2f})")
    ax.set_ylabel("feature dimension")
    ax.set_xlabel("time frame")
    fig.colorbar(im, ax=ax, label="feature value")
    return _finish(fig, save_path, dpi=100)


def generate_gt_masks(
    out_dir: str | os.PathLike | None = None,
    num_bins: int = config.N_FREQ_BINS,
    num_frames: int = config.N_TIME_FRAMES,
    sr: int = config.SAMPLING_RATE,
    hop_length: int = config.HOP_LENGTH,
):
    out_dir = Path(out_dir or (config.VIZ_DIR / "GT_masks"))
    out_dir.mkdir(parents=True, exist_ok=True)
    f_max = sr / 2

    for i in range(8):
        f_low, f_high = i * 1000, (i + 1) * 1000
        mask = np.zeros((num_bins, num_frames))
        idx_low = int((f_low / f_max) * num_bins)
        idx_high = int((f_high / f_max) * num_bins)
        mask[idx_low:idx_high, :] = 1.0
        plot_mask(
            mask, title=f"ground-truth mask {f_low}-{f_high} Hz",
            sr=sr, hop_length=hop_length,
            save_path=out_dir / f"gt_mask_{f_low}_{f_high}Hz.png",
        )

    total_duration = (num_frames * hop_length) / sr
    time_mask = np.zeros((num_bins, num_frames))
    start_f = int((3 / total_duration) * num_frames)
    end_f = min(int((5 / total_duration) * num_frames), num_frames)
    time_mask[:, start_f:end_f] = 1.0
    plot_mask(
        time_mask, title="ground-truth mask time 3-5 s",
        sr=sr, hop_length=hop_length, save_path=out_dir / "gt_mask_time_3-5s.png",
    )
    print(f"Ground-truth masks written to {out_dir}")
