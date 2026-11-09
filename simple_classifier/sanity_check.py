"""How often the simple detector calls each input "fake", on a sample of clips:
the fake clip, its bona fide clip, the ground-truth mask applied with zero
fill and with real fill, the bona fide clip with the ground-truth region (or
everything else) set to zero, and an all-zero input.

Also checks that ``MelLogRegDetector.contributions`` adds up to
logit(fake) - logit(bona fide).

Run from the repo root:

    CUDA_VISIBLE_DEVICES=0 python3 simple_classifier/sanity_check.py --n 200
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: config.py, audio.py, new_masking/

import argparse

import torch

import config
from audio import AudioProcessor
from common import (
    SETUP_NAMES, FakeAndRealDataset, collate_fake_and_real, complex_stft, detector_logits,
    detector_path, keep_masked_part, list_files, remove_masked_part,
)
from mel_detector import MelLogRegDetector
from metrics import GROUND_TRUTH

INPUTS = [
    "fake clip",
    "bona fide clip",
    "ground truth kept, zero fill",
    "ground truth removed, zero fill",
    "ground truth kept, real fill",
    "ground truth removed, real fill",
    "bona fide, zero outside ground truth",
    "bona fide, zero inside ground truth",
    "all zeros",
]


def load_batch(paths, ap, device):
    dataset = FakeAndRealDataset(paths, ap)
    items = []
    for i in range(len(dataset)):
        items.append(dataset[i])
    fakes, reals, paths = collate_fake_and_real(items)
    return fakes.to(device), reals.to(device), paths


def check_setup(ap, setup_name, n_files, device):
    setup = config.get_setup(setup_name)
    detector = MelLogRegDetector(detector_path(setup_name)).to(device)
    all_paths = list_files(setup)
    step = len(all_paths) // n_files
    paths = all_paths[::step][:n_files]

    called_fake = {}
    for name in INPUTS:
        called_fake[name] = 0
    largest_sum_error = 0.0

    for start in range(0, len(paths), 20):
        fakes, reals, batch_paths = load_batch(paths[start:start + 20], ap, device)
        with torch.no_grad():
            x_fake = complex_stft(ap, fakes)
            x_real = complex_stft(ap, reals)
            gt_list = []
            for path in batch_paths:
                gt_list.append(GROUND_TRUTH[setup.localization](path, x_fake.shape[1:]))
            gt = torch.stack(gt_list).to(device)

            spectrograms = {
                "fake clip": x_fake,
                "bona fide clip": x_real,
                "ground truth kept, zero fill": keep_masked_part(gt, x_fake, x_real, "zero"),
                "ground truth removed, zero fill": remove_masked_part(gt, x_fake, x_real, "zero"),
                "ground truth kept, real fill": keep_masked_part(gt, x_fake, x_real, "real"),
                "ground truth removed, real fill": remove_masked_part(gt, x_fake, x_real, "real"),
                "bona fide, zero outside ground truth": gt * x_real,
                "bona fide, zero inside ground truth": (1 - gt) * x_real,
                "all zeros": 0 * x_fake,
            }
            for name in INPUTS:
                logits = detector_logits(ap, detector, spectrograms[name])
                called_fake[name] += int((logits > 0).sum().item())

            # the contributions must add up to the logit difference
            fake_waveforms = ap.compute_invert_stft(x_fake)
            real_waveforms = ap.compute_invert_stft(x_real)
            difference = detector(fake_waveforms) - detector(real_waveforms)                   # (batch,)
            summed = detector.contributions(fake_waveforms, real_waveforms).sum(dim=(1, 2))    # (batch,)
            largest_sum_error = max(largest_sum_error, (difference - summed).abs().max().item())

    print(f"{setup_name}: share of {len(paths)} clips called fake")
    for name in INPUTS:
        print(f"   {name:40s} {called_fake[name] / len(paths):.2f}")
    print(f"   largest |logit difference - sum of contributions|: {largest_sum_error:.2e}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=200, help="clips per setup")
    args = p.parse_args()
    device = config.get_device()
    ap = AudioProcessor(device=device)
    for setup_name in SETUP_NAMES:
        check_setup(ap, setup_name, args.n, device)


if __name__ == "__main__":
    main()
