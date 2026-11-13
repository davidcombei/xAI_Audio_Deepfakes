

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from audio import AudioProcessor
from data import AudioDataset, FakeAndRealDataset, collate_fake_and_real, collate_features, read_paths
from explainers import build_mask_predictor
from masking import (FILLS, check_fill, fill_spectrogram, keep_masked_part, load_fill_reference,
                     remove_masked_part)
from metrics import average_drop, average_gain, average_increase, faithfulness, fidelity
from models import TorchLogReg


def _build(setup, batch_size, device):
    setup = config.get_setup(setup) if isinstance(setup, str) else setup
    device = config.get_device(device)
    ap = AudioProcessor(device=device)
    loader = DataLoader(
        AudioDataset(read_paths(setup.filelist), ap, root=config.PROJECT_ROOT, device=device),
        batch_size=batch_size, shuffle=False,
        collate_fn=partial(collate_features, audio_processor=ap),
    )
    predict_mask = build_mask_predictor(setup, device)
    return setup, ap, loader, predict_mask, device


@torch.no_grad()
def export_npy(setup, batch_size=4, device=None):
    setup, ap, loader, predict_mask, _ = _build(setup, batch_size, device)
    key = config.setup_key(setup)
    spec_dir = config.NPY_DIR / "spectrograms" / key
    mask_dir = config.NPY_DIR / "masks" / key
    spec_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for _, magnitude, phase, paths in tqdm(loader, desc="export npy", ascii=True):
        mask = predict_mask(magnitude, paths)
        magnitude = magnitude.cpu().numpy()
        mask = mask.cpu().numpy()
        for i, path in enumerate(paths):
            name = Path(path).name
            np.save(spec_dir / f"{name}_spectrogram.npy", magnitude[i])
            np.save(mask_dir / f"{name}_mask.npy", mask[i])
    print(f"Saved spectrograms -> {spec_dir}\nSaved masks -> {mask_dir}")


@torch.no_grad()
def export_features(setup, batch_size=4, device=None, fill="real"):
    setup, ap, _, predict_mask, device = _build(setup, batch_size, device)
    check_fill(setup, fill)
    real_dir = setup.real_dir if fill == "real" else None
    reference = load_fill_reference(setup, fill, device)
    loader = DataLoader(
        FakeAndRealDataset(read_paths(setup.filelist), ap, root=config.PROJECT_ROOT, real_dir=real_dir),
        batch_size=batch_size, shuffle=False, collate_fn=collate_fake_and_real,
    )
    key = config.setup_key(setup)
    logreg = TorchLogReg(setup.detector_ckpt).to(device)
    out_dir = config.masked_features_dir(setup)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = {"original": [], "relevant": [], "irrelevant": []}
    preds, theta_out, masked_preds = [], [], []

    for fakes, reals, paths in tqdm(loader, desc=f"export features ({fill} fill)", ascii=True):
        fakes = fakes.to(device)                       # (batch, 80000)
        x_fake, magnitude, _ = ap.compute_stft(fakes)  # (batch, 513, 313)
        x_real, _, _ = ap.compute_stft(reals.to(device))
        x_real = fill_spectrogram(fill, reference, x_fake, x_real)
        mask = predict_mask(magnitude, paths)          # (batch, 513, 313)

        features = ap.extract_features(fakes)
        _, probs_clean = logreg(features.mean(dim=1))
        preds.append(probs_clean)
        feats["original"].append(features.mean(dim=1).cpu().numpy())

        for tag, spec, store in (
            ("relevant", keep_masked_part(mask, x_fake, x_real, fill), theta_out),
            ("irrelevant", remove_masked_part(mask, x_fake, x_real, fill), masked_preds),
        ):
            f = ap.extract_features(ap.compute_invert_stft(spec)).mean(dim=1)
            feats[tag].append(f.cpu().numpy())
            _, probs = logreg(f)
            store.append(probs)

    for tag, arrs in feats.items():
        np.save(out_dir / f"features_{tag}_{key}.npy", np.concatenate(arrs, axis=0))

    preds = torch.cat(preds)
    theta_out = torch.cat(theta_out)
    masked_preds = torch.cat(masked_preds)
    print(f"\nfeatures saved -> {out_dir}")
    print(f"faithfulness    : {faithfulness(preds, masked_preds).mean().item():.2f}")
    print(f"fidelity        : {fidelity(theta_out, preds).mean().item():.2f}")
    print(f"average drop    : {average_drop(theta_out, preds).mean().item():.2f}")
    print(f"average increase: {average_increase(theta_out, preds).mean().item():.2f}")
    print(f"average gain    : {average_gain(theta_out, preds).mean().item():.2f}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    for mode in ("npy", "features"):
        s = sub.add_parser(mode)
        s.add_argument("--setup", choices=config.SETUP_NAMES, required=True)
        s.add_argument("--batch-size", type=int, default=4)
        s.add_argument("--device", default=None)
        if mode == "features":
            s.add_argument("--fill", choices=FILLS, default="real",
                           help="real = masked-out bins filled from the bona fide clip; avg = from the setup's average spectrogram (compute_avg_spectrogram.py); avgreal = from the average bona fide spectrogram (compute_avg_spectrogram.py --source real); optimized = from the fill optimised to be neutral for the setup's detector (learn_neutral_fill.py); zero = original masking")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "npy":
        export_npy(args.setup, args.batch_size, args.device)
    else:
        export_features(args.setup, args.batch_size, args.device, args.fill)


if __name__ == "__main__":
    main()
