"""Train the log-mel logistic regression on real clips vs. ONE vocoded band (2-3 kHz).

Data (5 000 real + 5 000 fake, one of each per LJSpeech utterance):
  * real, label 0:  LJSpeech_vocoded22K/<id>.wav
  * fake, label 1:  --fakes rect   LJSpeech_freq_swap/<id>.wav_vocoded_2000-3000.wav
                                   (the original swaps: hard 2000-3000 Hz band)
                    --fakes mel    logReg_coeff_logMel/LJSpeech_freq_swap_mel/<id>.wav_vocoded_mel43-52.wav
                                   (made by make_mel_swaps.py: band = mel filters 43..52)
                    --fakes mel_hann  logReg_coeff_logMel/LJSpeech_freq_swap_mel_hann/...
                                   (same band, Hann-window swap: no leakage outside the band)

The detector is the one of simple_classifier/ (mel_detector.py):

    waveform (5 s, 16 kHz) -> 80-band log-mel -> mean over time (80 numbers)
      -> standardise -> logistic regression (80 coefficients + 1 bias)

Each clip first goes through the pipeline's STFT -> ISTFT round trip, because
that is how the detector sees clips when the explanations are computed.

The utterances are split 80 / 20 (the real and the fake clip of one utterance
stay on the same side), like simple_classifier/train_detector.py.

Besides accuracy and EER, the script reports how much of the detector's
decision comes from the swapped mel bands 43..52. The logit is linear in the
features, so for a fake and its real twin

    logit(fake) - logit(real) = sum over bands k of  coef_k / std_k * (feature_fake[k] - feature_real[k])

and each term is the exact contribution of mel band k.

The checkpoint has the same format as simple_classifier/detectors/*.pt, so
MelLogRegDetector(checkpoint) loads it and the xAI pipeline can use it as is.

Run from the repo root:

    CUDA_VISIBLE_DEVICES=1 python3 logReg_coeff_logMel/train_detector.py --fakes rect
    CUDA_VISIBLE_DEVICES=1 python3 logReg_coeff_logMel/train_detector.py --fakes mel
    CUDA_VISIBLE_DEVICES=1 python3 logReg_coeff_logMel/train_detector.py --fakes mel_hann
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))                          # config.py, audio.py, new_masking/
sys.path.insert(0, str(REPO_ROOT / "simple_classifier"))    # common.py, mel_detector.py

import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from audio import AudioProcessor
from common import FakeAndRealDataset, collate_fake_and_real, one_thread_per_worker, stft_round_trip, utterance_id
from mel_detector import MelLogRegDetector
from make_mel_swaps import OUT_DIRS as MEL_SWAP_DIRS
from make_mel_swaps import swapped_mel_bands

EXPERIMENT_DIR = Path(__file__).resolve().parent
DETECTOR_DIR = EXPERIMENT_DIR / "detectors"
FAKE_KINDS = ["rect", "mel", "mel_hann"]


def list_fakes(fakes):
    """Repo-relative paths of the 5 000 fake clips of one kind."""
    if fakes == "rect":
        folder = config.SETUPS["freq-swapped"].data_dir
        pattern = "*.wav_vocoded_2000-3000.wav"
    elif fakes == "mel":
        folder = MEL_SWAP_DIRS["rect"]
        pattern = "*.wav_vocoded_mel*.wav"
    else:
        folder = MEL_SWAP_DIRS["hann"]
        pattern = "*.wav_vocoded_mel*.wav"
    paths = []
    for wav in sorted(folder.glob(pattern)):
        paths.append(str(wav.relative_to(config.PROJECT_ROOT)))
    return paths


def compute_features(paths, ap, detector, device, batch_size, workers):
    """Time-averaged log-mel of every fake clip and of its real twin.

    Returns fake_features (n, 80), real_features (n, 80), ids (n,) - row i of both is the same utterance.
    """
    loader = DataLoader(
        FakeAndRealDataset(paths, ap), batch_size=batch_size,
        collate_fn=collate_fake_and_real, num_workers=workers, worker_init_fn=one_thread_per_worker,
    )
    fake_features = []
    real_features = []
    ids = []
    for fakes, reals, batch_paths in tqdm(loader, desc="features", ascii=True, mininterval=30):
        with torch.no_grad():
            fake_batch = detector.features(stft_round_trip(ap, fakes.to(device)))  # (batch, 80)
            real_batch = detector.features(stft_round_trip(ap, reals.to(device)))  # (batch, 80)
        for i in range(len(batch_paths)):
            fake_features.append(fake_batch[i].cpu().numpy())
            real_features.append(real_batch[i].cpu().numpy())
            ids.append(utterance_id(batch_paths[i]))
    return np.array(fake_features), np.array(real_features), ids


def equal_error_rate(labels, scores):
    """EER: the error rate where false-accept rate = false-reject rate."""
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    false_negative_rate = 1 - true_positive_rate
    best = np.argmin(np.abs(false_negative_rate - false_positive_rate))
    return (false_positive_rate[best] + false_negative_rate[best]) / 2


def train(fakes, C, batch_size, workers, reuse_features, device):
    DETECTOR_DIR.mkdir(parents=True, exist_ok=True)
    feature_file = DETECTOR_DIR / f"features_{fakes}.npz"

    # 1. features (or the ones saved by an earlier run, with --reuse-features)
    if reuse_features and feature_file.exists():
        saved = np.load(feature_file)
        fake_features = saved["fake_features"]
        real_features = saved["real_features"]
        ids = []
        for uid in saved["ids"]:
            ids.append(str(uid))  # numpy string -> plain Python string
        print(f"loaded features from {feature_file}")
    else:
        paths = list_fakes(fakes)
        print(f"{len(paths)} fake clips ({fakes}) + their real twins")
        ap = AudioProcessor(device=device)
        detector = MelLogRegDetector().to(device)  # untrained: only its feature extraction is used here
        fake_features, real_features, ids = compute_features(paths, ap, detector, device, batch_size, workers)
        np.savez(feature_file, fake_features=fake_features, real_features=real_features, ids=np.array(ids))

    # 2. split the utterances 80 / 20 (same split for both kinds of fakes)
    all_ids = sorted(ids)
    rng = np.random.RandomState(0)
    rng.shuffle(all_ids)
    n_test = len(all_ids) // 5
    test_ids = set(all_ids[:n_test])

    X_train, y_train, X_test, y_test = [], [], [], []
    test_rows = []  # rows (utterances) on the test side, for the contributions below
    for i in range(len(ids)):
        if ids[i] in test_ids:
            X_test.append(fake_features[i])
            y_test.append(1)
            X_test.append(real_features[i])
            y_test.append(0)
            test_rows.append(i)
        else:
            X_train.append(fake_features[i])
            y_train.append(1)
            X_train.append(real_features[i])
            y_train.append(0)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    # 3. standardise each mel band with the training mean / std, fit the logistic regression
    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0) + 1e-8
    model = LogisticRegression(C=C, max_iter=10000)
    model.fit((X_train - feature_mean) / feature_std, y_train)
    coefficients = model.coef_[0]                     # (80,) on standardised features

    # 4. test accuracy (on utterances the detector never saw)
    test_logits = model.decision_function((X_test - feature_mean) / feature_std)
    fakes_called_fake = np.mean(test_logits[y_test == 1] > 0)
    reals_called_real = np.mean(test_logits[y_test == 0] <= 0)
    eer = equal_error_rate(y_test, test_logits)

    # 5. exact contribution of each mel band to logit(fake) - logit(real), test utterances
    band_weights = coefficients / feature_std          # (80,) logit change per unit of mean log-mel
    contributions = []                                 # (n_test, 80)
    for i in test_rows:
        contributions.append(band_weights * (fake_features[i] - real_features[i]))
    mean_contribution = np.mean(np.array(contributions), axis=0)  # (80,)

    swapped = swapped_mel_bands()                      # 43..52
    inside = 0.0
    outside = 0.0
    for k in range(config.N_MELS):
        if k in swapped:
            inside += mean_contribution[k]
        else:
            outside += mean_contribution[k]

    report = (
        f"fakes = {fakes}: trained on {len(y_train)} clips ({int(y_train.sum())} fake), "
        f"tested on {len(y_test)} clips ({int(y_test.sum())} fake) from {n_test} held-out utterances, C={C}\n"
        f"  test: fakes called fake {fakes_called_fake:.3f}, real called real {reals_called_real:.3f}, "
        f"balanced accuracy {(fakes_called_fake + reals_called_real) / 2:.3f}, EER {eer * 100:.2f} %\n"
        f"  mean logit(fake) - logit(real) = {inside + outside:.3f}: "
        f"{inside:.3f} from mel bands {swapped[0]}-{swapped[-1]} (2-3 kHz), {outside:.3f} from the other 70 bands"
    )
    print(report)

    out_path = DETECTOR_DIR / f"mel_logreg_{fakes}.pt"
    torch.save({
        "feature_mean": torch.tensor(feature_mean, dtype=torch.float32),
        "feature_std": torch.tensor(feature_std, dtype=torch.float32),
        "weight": torch.tensor(coefficients, dtype=torch.float32),
        "bias": torch.tensor(model.intercept_, dtype=torch.float32),
        "test_ids": sorted(test_ids),
        "mean_contribution": torch.tensor(mean_contribution, dtype=torch.float32),
        "report": report,
    }, out_path)
    (DETECTOR_DIR / f"mel_logreg_{fakes}.txt").write_text(report + "\n")
    print(f"saved -> {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fakes", choices=FAKE_KINDS, required=True)
    p.add_argument("--C", type=float, default=1, help="inverse regularisation strength of the logistic regression")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=16, help="DataLoader workers")
    p.add_argument("--reuse-features", action="store_true", help="refit from the features saved by an earlier run")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    train(args.fakes, args.C, args.batch_size, args.workers, args.reuse_features, config.get_device(args.device))


if __name__ == "__main__":
    main()
