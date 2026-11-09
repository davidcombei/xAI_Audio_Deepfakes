"""Figures for the simple-classifier experiment, written to simple_classifier/results/figures/.

  detector_weights.png    how much each mel band moves the logit, for both detectors
  mask_profiles_freq.png  masks averaged over time, per swapped band
  mask_profiles_time.png  masks averaged over frequency (clips whose original is >= 5 s)
  examples.png            |X_fake - X_real| and the three masks, for 3 clips
  training_curves.png     loss terms per epoch of the 4 U-Net runs

Run from the repo root:

    CUDA_VISIBLE_DEVICES=0 python3 simple_classifier/figures.py \\
        --freq-zerofill-ckpt <.pth> --freq-realfill-ckpt <.pth> \\
        --time-zerofill-ckpt <.pth> --time-realfill-ckpt <.pth>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: config.py, audio.py, new_masking/

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader

import config
from audio import AudioProcessor
from common import (
    RESULTS_DIR, RUNS_DIR, FakeAndRealDataset, collate_fake_and_real, complex_stft, detector_path,
    list_files, real_path,
)
from mel_detector import MelLogRegDetector
from models import load_mask_predictor

FIGURE_DIR = RESULTS_DIR / "figures"

# Colours: categorical slots 1-4 of the dataviz reference palette (validated), grey text and grid.
BLUE = "#2a78d6"     # U-Net, zero fill
ORANGE = "#eb6834"   # U-Net, real fill
AQUA = "#1baf7a"     # classifier evidence
GT_SHADE = "#e4e3df" # ground-truth region (shaded, not a line)
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
MASK_COLORS = LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])
GREYS = LinearSegmentedColormap.from_list("greys", ["#fcfcfb", "#0b0b0b"])

METHODS = ["unet-zerofill", "unet-realfill", "classifier-evidence"]
LABELS = {
    "unet-zerofill": "U-Net, zero fill",
    "unet-realfill": "U-Net, real fill",
    "classifier-evidence": "classifier evidence",
}
COLORS = {"unet-zerofill": BLUE, "unet-realfill": ORANGE, "classifier-evidence": AQUA}

FREQUENCIES_KHZ = np.arange(config.N_FREQ_BINS) * (config.SAMPLING_RATE / 2) / config.N_FREQ_BINS / 1000  # (512,)
TIMES_S = np.arange(config.N_TIME_FRAMES) * config.HOP_LENGTH / config.SAMPLING_RATE                     # (248,)

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.titlecolor": INK, "axes.titlesize": 9, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2, "lines.solid_capstyle": "round",
    "figure.facecolor": "white", "savefig.dpi": 150, "savefig.bbox": "tight",
})


def all_masks(ap, detector, unets, x_fake, x_real):
    """The three masks of a batch, as numpy arrays (batch, 512, 248)."""
    masks = {}
    with torch.no_grad():
        masks["unet-zerofill"] = unets["zero"](x_fake.abs().unsqueeze(1)).squeeze(1).cpu().numpy()
        masks["unet-realfill"] = unets["real"](x_fake.abs().unsqueeze(1)).squeeze(1).cpu().numpy()
        masks["classifier-evidence"] = detector.evidence_mask(
            ap.compute_invert_stft(x_fake), ap.compute_invert_stft(x_real)).cpu().numpy()
    return masks


def load_models(setup_name, zerofill_ckpt, realfill_ckpt, device):
    detector = MelLogRegDetector(detector_path(setup_name)).to(device)
    unets = {
        "zero": load_mask_predictor(zerofill_ckpt, device),
        "real": load_mask_predictor(realfill_ckpt, device),
    }
    return detector, unets


def detector_weights(device):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True, sharey=True)
    setups = ["freq-swapped", "time-swapped"]
    for col in range(2):
        detector = MelLogRegDetector(detector_path(setups[col])).to(device)
        filters = detector.mel.mel_scale.fb.cpu().numpy()                     # (513, 80)
        centres_khz = filters.argmax(axis=0) * (config.SAMPLING_RATE / 2) / config.N_FREQ_BINS / 1000
        weights = detector.band_weights().cpu().numpy()                      # (80,)
        ax = axes[col]
        ax.axhline(0, color=INK_2, linewidth=1)
        ax.plot(centres_khz, weights, color=BLUE, marker="o", markersize=3)
        ax.set_title(f"{setups[col]} detector", loc="left")
        ax.set_xlabel("mel band centre (kHz)")
        if col == 0:
            ax.set_ylabel("logit change per unit\nof mean log-mel (> 0: fake)")
    fig.savefig(FIGURE_DIR / "detector_weights.png")
    plt.close(fig)


def mask_profiles_freq(ap, detector, unets, device, clips_per_band):
    setup = config.get_setup("freq-swapped")
    paths = list_files(setup)
    bands = []
    for start in range(0, 8000, 1000):
        bands.append((start, start + 1000))

    fig, axes = plt.subplots(2, 4, figsize=(12, 5), constrained_layout=True, sharex=True, sharey=True)
    for b in range(len(bands)):
        low, high = bands[b]
        band_paths = []
        for path in paths:
            if path.endswith(f"_{low}-{high}.wav"):
                band_paths.append(path)
        band_paths = band_paths[:clips_per_band]

        sums = {}
        for method in METHODS:
            sums[method] = np.zeros(config.N_FREQ_BINS)
        loader = DataLoader(FakeAndRealDataset(band_paths, ap), batch_size=16,
                            collate_fn=collate_fake_and_real, num_workers=4)
        for fakes, reals, _ in loader:
            with torch.no_grad():
                x_fake = complex_stft(ap, fakes.to(device))
                x_real = complex_stft(ap, reals.to(device))
            masks = all_masks(ap, detector, unets, x_fake, x_real)
            for method in METHODS:
                sums[method] += masks[method].mean(axis=2).sum(axis=0)  # sum over clips of the time-averaged mask

        ax = axes[b // 4, b % 4]
        ax.axvspan(low / 1000, high / 1000, color=GT_SHADE, linewidth=0)
        for method in METHODS:
            ax.plot(FREQUENCIES_KHZ, sums[method] / len(band_paths), color=COLORS[method], label=LABELS[method])
        ax.set_title(f"swapped band {low // 1000}-{high // 1000} kHz", loc="left")
        if b // 4 == 1:
            ax.set_xlabel("frequency (kHz)")
        if b % 4 == 0:
            ax.set_ylabel("mask, mean over time")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle(f"freq-swapped: masks averaged over time and {clips_per_band} clips per band "
                 "(grey: swapped band)", x=0.01, ha="left", color=INK)
    fig.savefig(FIGURE_DIR / "mask_profiles_freq.png")
    plt.close(fig)


def mask_profiles_time(ap, detector, unets, device, n_clips):
    setup = config.get_setup("time-swapped")
    long_paths = []
    for path in list_files(setup):
        if sf.info(str(real_path(path))).duration >= 5.0:  # 3-5 s holds vocoded speech
            long_paths.append(path)
        if len(long_paths) == n_clips:
            break

    sums = {}
    for method in METHODS:
        sums[method] = np.zeros(config.N_TIME_FRAMES)
    loader = DataLoader(FakeAndRealDataset(long_paths, ap), batch_size=16,
                        collate_fn=collate_fake_and_real, num_workers=4)
    for fakes, reals, _ in loader:
        with torch.no_grad():
            x_fake = complex_stft(ap, fakes.to(device))
            x_real = complex_stft(ap, reals.to(device))
        masks = all_masks(ap, detector, unets, x_fake, x_real)
        for method in METHODS:
            sums[method] += masks[method].mean(axis=1).sum(axis=0)  # sum over clips of the frequency-averaged mask

    fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
    ax.axvspan(3, 5, color=GT_SHADE, linewidth=0)
    for method in METHODS:
        ax.plot(TIMES_S, sums[method] / len(long_paths), color=COLORS[method], label=LABELS[method])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mask, mean over frequency")
    ax.legend(loc="upper left")
    ax.set_title(f"time-swapped: {len(long_paths)} clips whose original is >= 5 s (grey: swapped 3-5 s)", loc="left")
    fig.savefig(FIGURE_DIR / "mask_profiles_time.png")
    plt.close(fig)


def examples(ap, models, device):
    """One row per clip: |X_fake - X_real| and the three masks."""
    freq_paths = list_files(config.get_setup("freq-swapped"))
    time_paths = list_files(config.get_setup("time-swapped"))
    utterance = Path(time_paths[0]).name[:10]
    rows = []
    for path in freq_paths:
        if Path(path).name.startswith(utterance) and (path.endswith("_1000-2000.wav") or path.endswith("_6000-7000.wav")):
            rows.append(("freq-swapped", path))
    rows.append(("time-swapped", time_paths[0]))

    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 2.6 * len(rows)), constrained_layout=True)
    extent = [0, TIMES_S[-1], 0, FREQUENCIES_KHZ[-1]]
    for r in range(len(rows)):
        setup_name, path = rows[r]
        detector, unets = models[setup_name]
        fakes, reals, _ = collate_fake_and_real([FakeAndRealDataset([path], ap)[0]])
        with torch.no_grad():
            x_fake = complex_stft(ap, fakes.to(device))
            x_real = complex_stft(ap, reals.to(device))
        masks = all_masks(ap, detector, unets, x_fake, x_real)
        difference = np.log1p((x_fake - x_real).abs()[0].cpu().numpy())

        images = [(difference, GREYS, "log(1 + |X_fake - X_real|)")]
        for method in METHODS:
            images.append((masks[method][0], MASK_COLORS, LABELS[method]))
        for c in range(4):
            image, colormap, title = images[c]
            ax = axes[r, c]
            ax.imshow(image, origin="lower", aspect="auto", cmap=colormap, extent=extent,
                      vmin=0, vmax=image.max() if c == 0 else 1)
            ax.grid(False)
            ax.set_title(title if r == 0 else "", loc="left")
            if c == 0:
                if setup_name == "freq-swapped":
                    low, high = Path(path).stem.split("_")[-1].split("-")
                    swap = f"{int(low) // 1000}-{int(high) // 1000} kHz swapped"
                else:
                    swap = "3-5 s swapped"
                ax.set_ylabel(f"{Path(path).name[:10]}, {swap}\nfrequency (kHz)")
            if r == len(rows) - 1:
                ax.set_xlabel("time (s)")
    fig.savefig(FIGURE_DIR / "examples.png")
    plt.close(fig)


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


def training_curves():
    fig, axes = plt.subplots(2, 3, figsize=(10, 5), constrained_layout=True)
    setups = ["freq-swapped", "time-swapped"]
    fills = [("zero", BLUE), ("real", ORANGE)]
    for row in range(2):
        for fill, color in fills:
            epochs, l_in, l_out, l1 = read_loss_log(RUNS_DIR / f"{setups[row]}_{fill}fill" / "loss_terms.txt")
            curves = [(l_in, "l_in: keep mask -> same decision"), (l_out, "l_out: remove mask -> flips"),
                      (l1, "L1: mask mean")]
            for col in range(3):
                values, title = curves[col]
                ax = axes[row, col]
                ax.plot(epochs, values, color=color, label=f"U-Net, {fill} fill")
                ax.set_title(f"{setups[row]}: {title}", loc="left")
                ax.set_xlabel("epoch")
                if col == 1:
                    ax.set_yscale("log")
    axes[0, 0].legend()
    fig.savefig(FIGURE_DIR / "training_curves.png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--freq-zerofill-ckpt", required=True)
    p.add_argument("--freq-realfill-ckpt", required=True)
    p.add_argument("--time-zerofill-ckpt", required=True)
    p.add_argument("--time-realfill-ckpt", required=True)
    p.add_argument("--clips", type=int, default=200, help="clips per band (freq) / in total (time) for the profiles")
    args = p.parse_args()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    device = config.get_device()
    ap = AudioProcessor(device=device)

    models = {
        "freq-swapped": load_models("freq-swapped", args.freq_zerofill_ckpt, args.freq_realfill_ckpt, device),
        "time-swapped": load_models("time-swapped", args.time_zerofill_ckpt, args.time_realfill_ckpt, device),
    }
    training_curves()
    detector_weights(device)
    detector, unets = models["freq-swapped"]
    mask_profiles_freq(ap, detector, unets, device, args.clips)
    detector, unets = models["time-swapped"]
    mask_profiles_time(ap, detector, unets, device, args.clips)
    examples(ap, models, device)
    print(f"figures -> {FIGURE_DIR}")


if __name__ == "__main__":
    main()
