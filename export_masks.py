

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
from data import AudioDataset, collate_features, read_paths
from metrics import average_drop, average_gain, average_increase, faithfulness, fidelity
from models import TorchLogReg, load_mask_predictor


def _build(setup, batch_size, device):
    setup = config.get_setup(setup) if isinstance(setup, str) else setup
    device = config.get_device(device)
    ap = AudioProcessor(device=device)
    loader = DataLoader(
        AudioDataset(read_paths(setup.filelist), ap, root=config.PROJECT_ROOT, device=device),
        batch_size=batch_size, shuffle=False,
        collate_fn=partial(collate_features, audio_processor=ap),
    )
    model = load_mask_predictor(setup.mask_ckpt, device)
    return setup, ap, loader, model, device


@torch.no_grad()
def export_npy(setup, batch_size=4, device=None):
    setup, ap, loader, model, _ = _build(setup, batch_size, device)
    key = config.setup_key(setup)
    spec_dir = config.NPY_DIR / "spectrograms" / key
    mask_dir = config.NPY_DIR / "masks" / key
    spec_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for _, magnitude, phase, paths in tqdm(loader, desc="export npy", ascii=True):
        mask = model(magnitude.unsqueeze(1)).squeeze(1)
        Tmax = mask.shape[1]
        magnitude = magnitude[:, :Tmax, :].cpu().numpy()
        mask = mask.cpu().numpy()
        for i, path in enumerate(paths):
            name = Path(path).name
            np.save(spec_dir / f"{name}_spectrogram.npy", magnitude[i])
            np.save(mask_dir / f"{name}_mask.npy", mask[i])
    print(f"Saved spectrograms -> {spec_dir}\nSaved masks -> {mask_dir}")


@torch.no_grad()
def export_features(setup, batch_size=4, device=None):
    setup, ap, loader, model, device = _build(setup, batch_size, device)
    key = config.setup_key(setup)
    logreg = TorchLogReg(setup.detector_ckpt).to(device)
    out_dir = config.masked_features_dir(setup)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = {"original": [], "relevant": [], "irrelevant": []}
    preds, theta_out, masked_preds = [], [], []

    for features, magnitude, phase, _ in tqdm(loader, desc="export features", ascii=True):
        mask = model(magnitude.unsqueeze(1)).squeeze(1)
        Tmax = mask.shape[1]
        magnitude = magnitude[:, :Tmax, :].to(device)
        phase = phase[:, :Tmax, :].to(device)

        _, probs_clean = logreg(features.mean(dim=1))
        preds.append(probs_clean)
        feats["original"].append(features.mean(dim=1).cpu().numpy())

        for tag, masked_mag, store in (
            ("relevant", mask * magnitude, theta_out),
            ("irrelevant", (1 - mask) * magnitude, masked_preds),
        ):
            spec = masked_mag * torch.exp(1j * phase)
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
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "npy":
        export_npy(args.setup, args.batch_size, args.device)
    else:
        export_features(args.setup, args.batch_size, args.device)


if __name__ == "__main__":
    main()
