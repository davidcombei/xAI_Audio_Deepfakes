"""Checks run before training (results are in the README).

1. With fill="zero", the new loss gives the same numbers as the original
   losses.LMACLoss (full 48-layer wav2vec2, two separate passes).
2. How often the detector calls each input "fake", on a sample of clips:
   the fake clip, its bona fide clip, the ground-truth mask applied with zero
   fill and with real fill, and the bona fide clip with the ground-truth
   region (or everything else) set to zero.

Run from the repo root:

    CUDA_VISIBLE_DEVICES=0 python3 new_masking/sanity_check.py --n 60
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: config.py, audio.py, models.py

import argparse

import torch

import config
from audio import AudioProcessor
from common import (
    SETUP_NAMES, FakeAndRealDataset, collate_fake_and_real, complex_stft, detector_logits,
    keep_masked_part, keep_only_needed_wav2vec2_layers, list_files, remove_masked_part,
    use_math_attention,
)
from losses import LMACLoss
from metrics import GROUND_TRUTH
from models import TorchLogReg, UNet
from real_fill_loss import RealFillLMACLoss


def load_batch(paths, ap, device):
    dataset = FakeAndRealDataset(paths, ap)
    items = []
    for i in range(len(dataset)):
        items.append(dataset[i])
    fakes, reals, paths = collate_fake_and_real(items)
    return fakes.to(device), reals.to(device), paths


def check_loss_equivalence(ap, device):
    setup = config.get_setup("freq-swapped")
    logreg = TorchLogReg(setup.detector_ckpt).to(device)
    fakes, reals, _ = load_batch(list_files(setup)[:4], ap, device)
    x_fake = complex_stft(ap, fakes)
    x_real = complex_stft(ap, reals)
    torch.manual_seed(0)
    unet = UNet().to(device)
    with torch.no_grad():
        mask = unet(x_fake.abs().unsqueeze(1))
        target = torch.sigmoid(logreg(ap.extract_features(fakes).mean(1))[0])
        original_loss = LMACLoss(ap, logreg).to(device)
        old_total, old_parts, _ = original_loss.loss_function(mask, x_fake.abs(), x_fake.angle(), target)
        keep_only_needed_wav2vec2_layers(ap)   # the new code always runs with the cut wav2vec2
        new_loss = RealFillLMACLoss(ap, logreg, fill="zero").to(device)
        new_total, new_parts, _ = new_loss.loss_function(mask, x_fake, x_real, target)
    print("1. original LMACLoss      ", round(old_total.item(), 4), old_parts.tolist())
    print("   RealFillLMACLoss(zero) ", round(new_total.item(), 4), new_parts.tolist())


def check_detector_decisions(ap, setup_name, n_files, device):
    setup = config.get_setup(setup_name)
    logreg = TorchLogReg(setup.detector_ckpt).to(device)
    all_paths = list_files(setup)
    step = len(all_paths) // n_files
    paths = all_paths[::step][:n_files]

    inputs = [
        "fake clip",
        "bona fide clip",
        "ground truth kept, zero fill",
        "ground truth removed, zero fill",
        "ground truth kept, real fill",
        "ground truth removed, real fill",
        "bona fide, zero outside ground truth",
        "bona fide, zero inside ground truth",
    ]
    called_fake = {}
    for name in inputs:
        called_fake[name] = 0

    for start in range(0, len(paths), 6):
        fakes, reals, batch_paths = load_batch(paths[start:start + 6], ap, device)
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
            }
            for name in inputs:
                logits = detector_logits(ap, logreg, spectrograms[name])
                called_fake[name] += int((logits > 0).sum().item())

    print(f"2. {setup_name}: share of {len(paths)} clips called fake")
    for name in inputs:
        print(f"   {name:40s} {called_fake[name] / len(paths):.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=60, help="clips per setup for check 2")
    args = p.parse_args()
    use_math_attention()
    device = config.get_device()
    ap = AudioProcessor(device=device)

    check_loss_equivalence(ap, device)
    for setup_name in SETUP_NAMES:
        check_detector_decisions(ap, setup_name, args.n, device)


if __name__ == "__main__":
    main()
