"""Figures of this experiment (written to logReg_coeff_logMel/figures/).

1. swap_bands.png    what is replaced by vocoded content, original hard band vs.
                     mel-aligned band, drawn over the detector's mel filters
2. coefficients.png  the logistic-regression coefficient of every mel band, and
                     each band's mean contribution to logit(fake) - logit(real)
                     (held-out utterances), for the three detectors

Run from the repo root after train_detector.py:

    python3 logReg_coeff_logMel/figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: config.py

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import config
from make_mel_swaps import mel_points, swapped_mel_bands, triangle

EXPERIMENT_DIR = Path(__file__).resolve().parent
DETECTOR_DIR = EXPERIMENT_DIR / "detectors"
FIGURE_DIR = EXPERIMENT_DIR / "figures"

FAKE_KINDS = ["rect", "mel", "mel_hann"]
COLORS = {"rect": "#2a78d6", "mel": "#eb6834", "mel_hann": "#1baf7a"}
LABELS = {
    "rect": "original swap (hard 2000-3000 Hz)",
    "mel": "mel-aligned swap (filters 43-52)",
    "mel_hann": "mel-aligned swap, Hann window",
}
FILTER_GRAY = "#b5b4ad"
BAND_SHADE = "#ecebe7"
TEXT = "#52514e"


def style_axis(ax):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(FILTER_GRAY)
    ax.spines["bottom"].set_color(FILTER_GRAY)
    ax.tick_params(colors=TEXT)
    ax.grid(axis="y", color=BAND_SHADE, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_swap_bands():
    points = mel_points()
    swapped = swapped_mel_bands()
    frequencies = []
    for i in range(1601):
        frequencies.append(1700 + i * (3300 - 1700) / 1600)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    # the detector's mel filters around 2-3 kHz (darker: the 10 swapped ones)
    for k in range(38, 58):
        values = []
        for f in frequencies:
            values.append(triangle(f, points[k], points[k + 1], points[k + 2]))
        if k in swapped:
            ax.plot(frequencies, values, color=TEXT, linewidth=0.8)
        else:
            ax.plot(frequencies, values, color=FILTER_GRAY, linewidth=0.8)

    # original swap: every bin in [2000, 3000) Hz
    hard = []
    for f in frequencies:
        if 2000 <= f < 3000:
            hard.append(1.0)
        else:
            hard.append(0.0)
    ax.plot(frequencies, hard, color=COLORS["rect"], linewidth=2, label=LABELS["rect"])

    # mel-aligned swap: W(f) = sum of filters 43..52
    soft = []
    for f in frequencies:
        total = 0.0
        for k in swapped:
            total += triangle(f, points[k], points[k + 1], points[k + 2])
        soft.append(min(total, 1.0))
    ax.plot(frequencies, soft, color=COLORS["mel"], linewidth=2, linestyle="--",
            label="mel-aligned swap (filters 43-52, both windows)")

    ax.set_xlim(1700, 3300)
    ax.set_ylim(-0.02, 1.12)
    ax.set_xlabel("frequency (Hz)", color=TEXT)
    ax.set_ylabel("share taken from the vocoded clip", color=TEXT)
    ax.set_title("What is replaced: thin lines = detector's mel filters (dark = 43-52)", color=TEXT, fontsize=10)
    ax.legend(frameon=False, loc="upper center", fontsize=8, ncol=2, bbox_to_anchor=(0.5, -0.2))
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "swap_bands.png", dpi=150)
    plt.close(fig)


def plot_coefficients():
    points = mel_points()
    swapped = swapped_mel_bands()
    bands = list(range(config.N_MELS))

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax in axes:
        ax.axvspan(swapped[0] - 0.5, swapped[-1] + 0.5, color=BAND_SHADE, linewidth=0)
        ax.axhline(0, color=FILTER_GRAY, linewidth=0.8)

    for fakes in FAKE_KINDS:
        if not (DETECTOR_DIR / f"mel_logreg_{fakes}.pt").exists():
            continue
        checkpoint = torch.load(DETECTOR_DIR / f"mel_logreg_{fakes}.pt", map_location="cpu")
        coefficients = checkpoint["weight"].numpy()
        contributions = checkpoint["mean_contribution"].numpy()
        axes[0].plot(bands, coefficients, color=COLORS[fakes], linewidth=2, marker="o", markersize=3,
                     label=LABELS[fakes])
        axes[1].plot(bands, contributions, color=COLORS[fakes], linewidth=2, marker="o", markersize=3,
                     label=LABELS[fakes])

    axes[0].set_ylabel("coefficient\n(standardised feature)", color=TEXT)
    axes[0].set_title("Logistic-regression coefficients per mel band (shaded: swapped bands 43-52, 2-3 kHz)",
                      color=TEXT, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_ylabel("mean contribution to\nlogit(fake) - logit(real)", color=TEXT)
    axes[1].set_title("Where the decision comes from (held-out utterances, exact split of the logit)",
                      color=TEXT, fontsize=10)

    ticks = []
    tick_labels = []
    for k in range(0, config.N_MELS, 10):
        ticks.append(k)
        tick_labels.append(f"{k}\n{points[k + 1]:.0f} Hz")
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels(tick_labels)
    axes[1].set_xlabel("mel band (centre frequency)", color=TEXT)
    axes[1].set_xlim(-1, config.N_MELS)
    for ax in axes:
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "coefficients.png", dpi=150)
    plt.close(fig)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_swap_bands()
    plot_coefficients()
    print(f"figures -> {FIGURE_DIR}")


if __name__ == "__main__":
    main()
