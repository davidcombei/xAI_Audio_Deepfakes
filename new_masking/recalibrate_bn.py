"""Recompute the BatchNorm statistics of a trained U-Net (weights stay unchanged).

During training, BatchNorm normalises with the statistics of each batch and
keeps a running average (momentum 0.1) for evaluation. That running average
mostly reflects the last few batches of the run. For the freq-swapped U-Net
at epoch 40 it was off: in eval mode only 52 % of 240 sample clips flipped
when the mask was removed, against 98 % with batch statistics (sanity check in
the README). This script replaces the running averages with a plain average
over one pass of the setup's files, in batches of 15 like in training, and
saves the result next to the checkpoint as ``<name>_bn_recalibrated.pth``.

Run from the repo root:

    CUDA_VISIBLE_DEVICES=0 python3 new_masking/recalibrate_bn.py --setup freq-swapped \\
        --ckpt new_masking/runs/freq-swapped_realfill/UNet_MP_epoch_40_loss_0.4558.pth
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: config.py, audio.py, models.py

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from audio import AudioProcessor
from common import SETUP_NAMES, FakeAndRealDataset, collate_fake_and_real, complex_stft, list_files
from models import load_mask_predictor


def recalibrate(setup_name, ckpt_path, batch_size=15, device=None):
    setup = config.get_setup(setup_name)
    device = config.get_device(device)
    ap = AudioProcessor(device=device)
    unet = load_mask_predictor(ckpt_path, device)

    # Forget the running averages and switch to a plain (cumulative) average.
    for module in unet.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None

    loader = DataLoader(
        FakeAndRealDataset(list_files(setup), ap), batch_size=batch_size, shuffle=True,
        collate_fn=collate_fake_and_real, num_workers=8, generator=torch.Generator().manual_seed(0),
    )
    unet.train()  # train mode: BatchNorm uses and accumulates batch statistics
    with torch.no_grad():
        for fakes, reals, paths in tqdm(loader, desc="recalibrate BatchNorm", ascii=True, mininterval=30):
            magnitude = complex_stft(ap, fakes.to(device)).abs()   # (batch, 512, 248)
            unet(magnitude.unsqueeze(1))                           # only the statistics change

    ckpt_path = Path(ckpt_path)
    out_path = ckpt_path.with_name(ckpt_path.stem + "_bn_recalibrated.pth")
    torch.save(unet.state_dict(), out_path)
    print(f"saved -> {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setup", choices=SETUP_NAMES, required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    recalibrate(args.setup, args.ckpt, device=args.device)


if __name__ == "__main__":
    main()
