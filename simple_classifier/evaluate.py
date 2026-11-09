"""Evaluate explanations of the simple detector with zero fill and with real fill.

Masks:
  unet-zerofill        U-Net trained against the simple detector with zero fill (--zerofill-ckpt)
  unet-realfill        U-Net trained against the simple detector with real fill (--realfill-ckpt)
  classifier-evidence  the detector's own, exact explanation (MelLogRegDetector.evidence_mask)
  ground-truth         the known manipulated region (metrics.GROUND_TRUTH)

For every mask and every fill ("zero", "real"):
  Fidelity     = share of clips where keeping only the masked part keeps the detector's decision
  Faithfulness = mean of  p_fake(original) - p_fake(masked part removed)   (as evaluate.py)
And per mask (it does not depend on the fill):
  AUC          = ROC-AUC of the mask against the ground-truth region, averaged over clips

Each GPU process evaluates one shard of the files and writes the detector
logits of every clip to a CSV; --summarize-only merges the shards and prints
the tables. Run from the repo root:

    for i in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$i python3 simple_classifier/evaluate.py --setup freq-swapped \\
          --zerofill-ckpt <.pth> --realfill-ckpt <.pth> --shard $i --num-shards 8 &
    done; wait
    python3 simple_classifier/evaluate.py --setup freq-swapped --summarize-only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: config.py, audio.py, new_masking/

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from audio import AudioProcessor
from common import (
    FILLS, RESULTS_DIR, SETUP_NAMES, FakeAndRealDataset, collate_fake_and_real, complex_stft,
    detector_logits, detector_path, keep_masked_part, list_files, remove_masked_part,
)
from mel_detector import MelLogRegDetector
from metrics import GROUND_TRUTH
from models import load_mask_predictor

METHODS = ["unet-zerofill", "unet-realfill", "classifier-evidence", "ground-truth"]


def ground_truth_masks(setup, paths, shape, device):
    """Ground-truth mask of each clip: (batch, 512, 248), 1 = manipulated."""
    make_mask = GROUND_TRUTH[setup.localization]
    masks = []
    for path in paths:
        masks.append(make_mask(path, shape))
    return torch.stack(masks).to(device)


def evaluate_shard(setup, zerofill_ckpt, realfill_ckpt, out_dir, shard, num_shards, batch_size, device):
    ap = AudioProcessor(device=device)
    detector = MelLogRegDetector(detector_path(setup.name)).to(device)
    unet_zerofill = load_mask_predictor(zerofill_ckpt, device)
    unet_realfill = load_mask_predictor(realfill_ckpt, device)

    all_paths = list_files(setup)
    my_paths = all_paths[shard::num_shards]
    loader = DataLoader(FakeAndRealDataset(my_paths, ap), batch_size=batch_size,
                        collate_fn=collate_fake_and_real, num_workers=4)

    rows = []
    for fakes, reals, paths in tqdm(loader, desc=f"{setup.name} shard {shard}", ascii=True, mininterval=30):
        with torch.no_grad():
            x_fake = complex_stft(ap, fakes.to(device))   # (batch, 512, 248)
            x_real = complex_stft(ap, reals.to(device))   # (batch, 512, 248)
            gt = ground_truth_masks(setup, paths, x_fake.shape[1:], device)

            masks = {}
            masks["unet-zerofill"] = unet_zerofill(x_fake.abs().unsqueeze(1)).squeeze(1)
            masks["unet-realfill"] = unet_realfill(x_fake.abs().unsqueeze(1)).squeeze(1)
            masks["classifier-evidence"] = detector.evidence_mask(
                ap.compute_invert_stft(x_fake), ap.compute_invert_stft(x_real))
            masks["ground-truth"] = gt

            # one row per clip, filled in step by step
            batch_rows = []
            original_logits = detector_logits(ap, detector, x_fake).cpu().numpy()
            real_logits = detector_logits(ap, detector, x_real).cpu().numpy()
            for i in range(len(paths)):
                batch_rows.append({
                    "file": paths[i],
                    "logit_original": float(original_logits[i]),
                    "logit_bona_fide": float(real_logits[i]),
                })

            for method in METHODS:
                mask = masks[method]
                for fill in FILLS:
                    keep_logits = detector_logits(ap, detector, keep_masked_part(mask, x_fake, x_real, fill))
                    remove_logits = detector_logits(ap, detector, remove_masked_part(mask, x_fake, x_real, fill))
                    keep_logits = keep_logits.cpu().numpy()
                    remove_logits = remove_logits.cpu().numpy()
                    for i in range(len(paths)):
                        batch_rows[i][f"{method}_{fill}_logit_keep"] = float(keep_logits[i])
                        batch_rows[i][f"{method}_{fill}_logit_remove"] = float(remove_logits[i])

        gt_np = gt.cpu().numpy()
        for method in METHODS:
            mask_np = masks[method].cpu().numpy()
            for i in range(len(paths)):
                batch_rows[i][f"{method}_auc"] = roc_auc_score(gt_np[i].flatten(), mask_np[i].flatten())
        rows.extend(batch_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / f"per_file_shard{shard}of{num_shards}.csv", index=False)


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def scores(df, method):
    """Fidelity and Faithfulness for each fill, and AUC, of one mask method over the clips in df."""
    original = df["logit_original"].to_numpy()
    result = {"mask": method}
    for fill in FILLS:
        keep = df[f"{method}_{fill}_logit_keep"].to_numpy()
        remove = df[f"{method}_{fill}_logit_remove"].to_numpy()
        result[f"Fidelity ({fill} fill)"] = np.mean((keep > 0) == (original > 0))
        result[f"Faithfulness ({fill} fill)"] = np.mean(sigmoid(original) - sigmoid(remove))
    result["AUC"] = df[f"{method}_auc"].mean()
    return result


def band_of(path):
    """'..._vocoded_2000-3000.wav' -> '2000-3000'."""
    return Path(path).stem.split("_")[-1]


def summarize(setup, out_dir):
    shard_tables = []
    for shard_file in sorted(out_dir.glob("per_file_shard*.csv")):
        shard_tables.append(pd.read_csv(shard_file))
    df = pd.concat(shard_tables, ignore_index=True).sort_values("file")
    df.to_csv(out_dir / "per_file.csv", index=False)

    lines = []
    called_fake = np.mean(df["logit_original"] > 0)
    bona_fide_called_fake = np.mean(df["logit_bona_fide"] > 0)
    lines.append(f"{setup.name}: {len(df)} clips, {called_fake:.3f} of them called fake by the detector "
                 f"(their bona fide clips: {bona_fide_called_fake:.3f})")
    lines.append("")

    rows = []
    for method in METHODS:
        rows.append(scores(df, method))
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "summary.csv", index=False)
    lines.append(table.to_string(index=False, float_format="%.3f"))

    if setup.localization == "freq":
        bands = []
        for start in range(0, 8000, 1000):
            bands.append(f"{start}-{start + 1000}")
        band_per_clip = []
        for path in df["file"]:
            band_per_clip.append(band_of(path))
        df["band"] = band_per_clip

        rows = []
        for method in METHODS:
            for band in bands:
                row = {"band": band}
                row.update(scores(df[df["band"] == band], method))
                rows.append(row)
        per_band = pd.DataFrame(rows)
        per_band.to_csv(out_dir / "summary_per_band.csv", index=False)
        lines.append("")
        lines.append("per swapped band:")
        lines.append(per_band.to_string(index=False, float_format="%.3f"))

    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text + "\n")
    print(text)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setup", choices=SETUP_NAMES, required=True)
    p.add_argument("--zerofill-ckpt", default=None)
    p.add_argument("--realfill-ckpt", default=None)
    p.add_argument("--out-dir", default=None, help="default: simple_classifier/results/<setup>")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--summarize-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    setup = config.get_setup(args.setup)
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR / args.setup
    if not args.summarize_only:
        if args.zerofill_ckpt is None or args.realfill_ckpt is None:
            raise SystemExit("--zerofill-ckpt and --realfill-ckpt are required")
        device = config.get_device(args.device)
        evaluate_shard(setup, args.zerofill_ckpt, args.realfill_ckpt, out_dir,
                       args.shard, args.num_shards, args.batch_size, device)
        if args.num_shards > 1:
            return  # merge the shards later with --summarize-only
    summarize(setup, out_dir)


if __name__ == "__main__":
    main()
