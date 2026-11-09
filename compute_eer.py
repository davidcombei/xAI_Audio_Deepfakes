"""EER of a setup's detector on the clips the explanations are evaluated on.

evaluate.py only scores fake clips (the bona fide clips are used only to fill masked-out bins),
so it cannot give an EER. This script scores both classes:
  fakes  every clip in setup.filelist (the clips evaluate.py uses)
  reals  the bona fide clip each fake was made from (data.real_path), each one scored once
         (freq-swapped: 8 fakes per utterance share one bona fide clip)

Every clip goes through the same path as in evaluate.py: STFT -> ISTFT -> wav2vec2 -> logistic
regression. The EER is computed as in train_detector.py (fake = 1, real = 0).

Note: unlike train_detector.py, which reports the EER on a held-out 20 % split, these clips can
overlap the detector's training data (realistic: 26706 of the 27171 fakes are in
metadata/mlaad_selection_metadata.txt), so the EER can be optimistic.

    python compute_eer.py -s realistic --device cuda:0
"""

from __future__ import annotations

import csv
from pathlib import Path

import click
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from audio import AudioProcessor
from data import AudioDataset, one_thread_per_worker, read_paths, real_path
from detector import build_detector


def compute_eer(y_true, y_scores) -> float:
    """Same as train_detector.compute_eer (copied: train_detector.py does not import right now)."""
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    return float(brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0))


def detector_scores(paths, ap, detector, device, batch_size, workers, desc):
    """p(fake) of every clip, through the same STFT -> ISTFT path as evaluate.py."""
    loader = DataLoader(
        AudioDataset(paths, ap, root=config.PROJECT_ROOT, device="cpu"),
        batch_size=batch_size, num_workers=workers, worker_init_fn=one_thread_per_worker,
    )
    scores = []
    for waveforms, _ in tqdm(loader, desc=desc, ascii=True, mininterval=30):
        _, magnitude, phase = ap.compute_stft(waveforms.to(device))   # (batch, 513, 313)
        with torch.no_grad():
            scores.append(torch.sigmoid(detector(magnitude, phase)).cpu().numpy())
    return np.concatenate(scores)


@click.command()
@click.option("-s", "--setup", type=click.Choice(config.SETUP_NAMES), required=True)
@click.option("-b", "--batch-size", default=8, show_default=True)
@click.option("--device", default=None, help="e.g. cuda, cuda:1, cpu")
@click.option("--workers", default=4, show_default=True, help="DataLoader workers (audio loading)")
@click.option("--scores-csv", default=None, help="also save p(fake) of every clip here")
def main(setup, batch_size, device, workers, scores_csv):
    setup = config.get_setup(setup)
    if setup.real_dir is None:
        raise click.UsageError(f"setup {setup.name!r} has no bona fide clips (real_dir), so no EER")
    device = config.get_device(device)
    ap = AudioProcessor(device=device)
    detector = build_detector(setup, ap, device=device)

    fakes = read_paths(setup.filelist)
    reals = list(dict.fromkeys(str(real_path(p, setup.real_dir)) for p in fakes))  # unique, in order
    fake_scores = detector_scores(fakes, ap, detector, device, batch_size, workers, f"{setup.name} fakes")
    real_scores = detector_scores(reals, ap, detector, device, batch_size, workers, f"{setup.name} reals")

    labels = np.r_[np.ones(len(fakes)), np.zeros(len(reals))]
    eer = compute_eer(labels, np.r_[fake_scores, real_scores])
    fakes_called_fake = float(np.mean(fake_scores > 0.5))
    reals_called_real = float(np.mean(real_scores <= 0.5))

    if scores_csv:
        Path(scores_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(scores_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "label", "p_fake"])
            w.writerows([p, 1, f"{s:.6f}"] for p, s in zip(fakes, fake_scores))
            w.writerows([p, 0, f"{s:.6f}"] for p, s in zip(reals, real_scores))

    print("\ndetector:")
    print(f"  EER (%): {eer * 100:.3f}")
    print(f"  fakes called fake: {fakes_called_fake:.3f}")
    print(f"  reals called real: {reals_called_real:.3f}")
    print(f"  balanced accuracy: {(fakes_called_fake + reals_called_real) / 2:.3f}")
    print(f"  n fakes: {len(fakes)}")
    print(f"  n reals: {len(reals)}")


if __name__ == "__main__":
    main()
