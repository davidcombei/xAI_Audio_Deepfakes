"""Figures for the real-fill experiment, written to new_masking/results/figures/.

  training_curves.png     loss terms per epoch of the real-fill run
  examples_freq.png       |X_fake - X_real|, ground truth and both U-Net masks (one utterance, 4 bands)
  mask_profiles_freq.png  mask averaged over time, per swapped band

Run from the repo root:

    CUDA_VISIBLE_DEVICES=0 python3 new_masking/figures.py --freq-ckpt <.pth>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: config.py, audio.py, models.py

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader

import config
from audio import AudioProcessor
from common import (
    RESULTS_DIR, RUNS_DIR, FakeAndRealDataset, collate_fake_and_real, complex_stft, list_files,
    utterance_id,
)
from metrics import GROUND_TRUTH
from models import load_mask_predictor

FIGURE_DIR = RESULTS_DIR / "figures"

# Colours: categorical slots 1-3 of the dataviz reference palette, grey text and grid.
BLUE = "#2a78d6"     # new U-Net (real fill)
ORANGE = "#eb6834"   # original U-Net (zero fill)
AQUA = "#1baf7a"     # ground truth
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
MASK_COLORS = LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])
GREYS = LinearSegmentedColormap.from_list("greys", ["#fcfcfb", "#0b0b0b"])

NEW_LABEL = "U-Net, real fill (new)"
OLD_LABEL = "U-Net, zero fill (original)"

FREQUENCIES_HZ = np.arange(config.N_FREQ_BINS) * (config.SAMPLING_RATE / 2) / config.N_FREQ_BINS  # (512,)
TIMES_S = np.arange(config.N_TIME_FRAMES) * config.HOP_LENGTH / config.SAMPLING_RATE            # (248,)

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.titlecolor": INK, "axes.titlesize": 9, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2, "lines.solid_capstyle": "round",
    "figure.facecolor": "white", "savefig.dpi": 150, "savefig.bbox": "tight",
})


def read_loss_log(path):
    """Parse 'Epoch N: l_in=..., l_out=..., L1=..., weights=...' lines -> (epochs, l_in, l_out, l1)."""
    epochs, l_in, l_out, l1 = [], [], [], []
    for line in Path(path).read_text().splitlines():
        if not line.startswith("Epoch "):
            continue
        epoch_part, values_part = line.split(":", 1)
        epochs.append(int(epoch_part.split()[1]))
        for item in values_part.split(","):
            item = item.strip()
            if item.startswith("l_in="):
                l_in.append(float(item[len("l_in="):]))
            elif item.startswith("l_out="):
                l_out.append(float(item[len("l_out="):]))
            elif item.startswith("L1="):
                l1.append(float(item[len("L1="):]))
    return epochs, l_in, l_out, l1


def training_curves(all_files_from_epoch):
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.5), constrained_layout=True)
    setup_name = "freq-swapped"
    epochs, l_in, l_out, l1 = read_loss_log(RUNS_DIR / f"{setup_name}_realfill" / "loss_terms.txt")
    curves = [
        (l_in, "l_in", "keep mask -> same decision", False),
        (l_out, "l_out", "remove mask -> decision flips", True),
        (l1, "L1", "mask mean", False),
    ]
    for col in range(3):
        values, term, meaning, log_scale = curves[col]
        ax = axes[col]
        ax.plot(epochs, values, color=BLUE)
        ax.axvline(all_files_from_epoch, color=INK_2, linewidth=1)
        ax.set_title(f"{setup_name}: {term}", loc="left")
        ax.set_ylabel(meaning)
        ax.set_xlabel("epoch")
        if log_scale:
            ax.set_yscale("log")
    fig.suptitle("Real-fill training (grey line: from here on, all files are used)",
                 x=0.01, ha="left", color=INK)
    fig.savefig(FIGURE_DIR / "training_curves.png")
    plt.close(fig)


def predict_masks(unet, x_fake):
    """(batch, 512, 248) complex STFT -> (batch, 512, 248) numpy mask."""
    with torch.no_grad():
        mask = unet(x_fake.abs().unsqueeze(1)).squeeze(1)
    return mask.cpu().numpy()


def examples(setup_name, realfill_ckpt, paths, out_name, device):
    setup = config.get_setup(setup_name)
    ap = AudioProcessor(device=device)
    unet_new = load_mask_predictor(realfill_ckpt, device)
    unet_old = load_mask_predictor(setup.mask_ckpt, device)

    dataset = FakeAndRealDataset(paths, ap)
    items = []
    for i in range(len(dataset)):
        items.append(dataset[i])
    fakes, reals, paths = collate_fake_and_real(items)
    x_fake = complex_stft(ap, fakes.to(device))
    x_real = complex_stft(ap, reals.to(device))
    difference = (x_fake - x_real).abs().cpu().numpy()
    new_masks = predict_masks(unet_new, x_fake)
    old_masks = predict_masks(unet_old, x_fake)

    titles = ["|X_fake - X_real| (log scale)", "ground truth", OLD_LABEL, NEW_LABEL]
    extent = [0, TIMES_S[-1], 0, config.SAMPLING_RATE / 2]
    fig, axes = plt.subplots(len(paths), 4, figsize=(12, 2.3 * len(paths)), sharex=True, sharey=True,
                             constrained_layout=True, squeeze=False)
    for row in range(len(paths)):
        ground_truth = GROUND_TRUTH[setup.localization](paths[row], x_fake.shape[1:]).numpy()
        images = [np.log10(difference[row] + 1e-4), ground_truth, old_masks[row], new_masks[row]]
        for col in range(4):
            ax = axes[row, col]
            if col == 0:
                ax.imshow(images[col], origin="lower", aspect="auto", extent=extent, cmap=GREYS)
            else:
                mask_image = ax.imshow(images[col], origin="lower", aspect="auto", extent=extent,
                                       cmap=MASK_COLORS, vmin=0, vmax=1, interpolation="nearest")
            ax.grid(False)
            if row == 0:
                ax.set_title(titles[col], loc="left")
        clip_name = Path(paths[row]).name.replace(".wav_vocoded_", "  band ").replace(".wav", "")
        axes[row, 0].set_ylabel(f"{clip_name}\nfrequency (Hz)", fontsize=8)
    for col in range(4):
        axes[-1, col].set_xlabel("time (s)")
    fig.colorbar(mask_image, ax=axes[:, 1:], shrink=0.6, label="mask value")
    fig.savefig(FIGURE_DIR / out_name)
    plt.close(fig)


def add_legend(fig, ax, ground_truth_label):
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color=AQUA, alpha=0.15, linewidth=0))
    labels.append(ground_truth_label)
    fig.legend(handles, labels, loc="outside lower left", ncol=3)


def mask_profiles_freq(realfill_ckpt, device, n_utterances=100):
    """Mean mask value per frequency bin, averaged over time and clips, for each swapped band."""
    setup = config.get_setup("freq-swapped")
    ap = AudioProcessor(device=device)
    unet_new = load_mask_predictor(realfill_ckpt, device)
    unet_old = load_mask_predictor(setup.mask_ckpt, device)

    all_paths = list_files(setup)
    utterances = sorted(set(utterance_id(p) for p in all_paths))[::50][:n_utterances]
    paths = []
    for path in all_paths:
        if utterance_id(path) in utterances:
            paths.append(path)

    bands = []
    for start in range(0, 8000, 1000):
        bands.append(f"{start}-{start + 1000}")
    new_sum = {}
    old_sum = {}
    count = {}
    for band in bands:
        new_sum[band] = np.zeros(config.N_FREQ_BINS)
        old_sum[band] = np.zeros(config.N_FREQ_BINS)
        count[band] = 0

    loader = DataLoader(FakeAndRealDataset(paths, ap), batch_size=16, collate_fn=collate_fake_and_real, num_workers=4)
    for fakes, _, batch_paths in loader:
        x_fake = complex_stft(ap, fakes.to(device))
        new_masks = predict_masks(unet_new, x_fake)   # (batch, 512, 248)
        old_masks = predict_masks(unet_old, x_fake)
        for i in range(len(batch_paths)):
            band = Path(batch_paths[i]).stem.split("_")[-1]
            new_sum[band] += new_masks[i].mean(axis=1)
            old_sum[band] += old_masks[i].mean(axis=1)
            count[band] += 1

    fig, axes = plt.subplots(2, 4, figsize=(11, 5), sharex=True, sharey=True, constrained_layout=True)
    for k in range(8):
        band = bands[k]
        ax = axes[k // 4, k % 4]
        low, high = band.split("-")
        ax.axvspan(int(low), int(high), color=AQUA, alpha=0.15, linewidth=0)
        ax.plot(FREQUENCIES_HZ, new_sum[band] / count[band], color=BLUE, label=NEW_LABEL)
        ax.plot(FREQUENCIES_HZ, old_sum[band] / count[band], color=ORANGE, label=OLD_LABEL)
        ax.set_title(f"swapped band {int(low) / 1000:g}-{int(high) / 1000:g} kHz", loc="left")
        ax.set_ylim(0, 1)
    for col in range(4):
        axes[1, col].set_xlabel("frequency (Hz)")
    for row in range(2):
        axes[row, 0].set_ylabel("mean mask value")
    add_legend(fig, axes[0, 0], "swapped band (ground truth)")
    fig.suptitle(f"Mask averaged over time and {len(utterances)} utterances per band", x=0.01, ha="left", color=INK)
    fig.savefig(FIGURE_DIR / "mask_profiles_freq.png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--freq-ckpt", required=True)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    device = config.get_device(args.device)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    training_curves(24)  # all files are used from epoch 24 on

    utterance = utterance_id(list_files(config.get_setup("freq-swapped"), "subset")[0])
    freq_paths = []
    for band in ["0-1000", "2000-3000", "4000-5000", "6000-7000"]:
        freq_paths.append(f"LJSpeech_freq_swap/{utterance}.wav_vocoded_{band}.wav")
    examples("freq-swapped", args.freq_ckpt, freq_paths, "examples_freq.png", device)

    mask_profiles_freq(args.freq_ckpt, device)
    print(f"figures -> {FIGURE_DIR}")


if __name__ == "__main__":
    main()
