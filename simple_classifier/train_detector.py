"""Train the simple detector (mel_detector.py) for one swap setup.

Data: every fake clip of the setup (label 1) and the bona fide clip each one
was made from (label 0, ``LJSpeech_vocoded22K/<id>.wav``). For the
freq-swapped setup that is 40 000 fakes (8 bands x 5 000 utterances) against
5 000 bona fide clips, so the classes are re-weighted to count equally.

Each clip goes through the pipeline's STFT -> ISTFT round trip first, because
that is how the detector sees clips when the explanations are computed (the
STFT crop drops the Nyquist bin and the last frame, see iSTFT_check/).

The utterances are split 80 / 20 (all versions of one utterance stay on the
same side). The logistic regression is fitted on the 80 % and its accuracy is
reported on the 20 %, like the repo's train_detector.py.

Run from the repo root:

    CUDA_VISIBLE_DEVICES=0 python3 simple_classifier/train_detector.py --setup freq-swapped
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: config.py, audio.py, new_masking/

import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from audio import AudioProcessor
from common import (
    DETECTOR_DIR, SETUP_NAMES, FakeAndRealDataset, collate_fake_and_real, detector_path,
    list_files, one_thread_per_worker, stft_round_trip, utterance_id,
)
from mel_detector import MelLogRegDetector


def compute_features(setup, ap, detector, device, batch_size, workers):
    """Time-averaged log-mel of every fake clip and of every bona fide clip.

    Returns fake_features (n_fakes, 80), fake_ids, real_features (n_utterances, 80), real_ids.
    """
    loader = DataLoader(
        FakeAndRealDataset(list_files(setup), ap), batch_size=batch_size,
        collate_fn=collate_fake_and_real, num_workers=workers, worker_init_fn=one_thread_per_worker,
    )
    fake_features = []
    fake_ids = []
    real_feature_of_id = {}  # the freq setup has 8 fakes per utterance; keep each bona fide clip once
    for fakes, reals, paths in tqdm(loader, desc=f"features {setup.name}", ascii=True, mininterval=30):
        with torch.no_grad():
            fake_batch = detector.features(stft_round_trip(ap, fakes.to(device)))  # (batch, 80)
            real_batch = detector.features(stft_round_trip(ap, reals.to(device)))  # (batch, 80)
        for i in range(len(paths)):
            uid = utterance_id(paths[i])
            fake_features.append(fake_batch[i].cpu().numpy())
            fake_ids.append(uid)
            real_feature_of_id[uid] = real_batch[i].cpu().numpy()

    real_ids = sorted(real_feature_of_id)
    real_features = []
    for uid in real_ids:
        real_features.append(real_feature_of_id[uid])
    return np.array(fake_features), fake_ids, np.array(real_features), real_ids


def equal_error_rate(labels, scores):
    """EER: the error rate where false-accept rate = false-reject rate."""
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    false_negative_rate = 1 - true_positive_rate
    best = np.argmin(np.abs(false_negative_rate - false_positive_rate))
    return (false_positive_rate[best] + false_negative_rate[best]) / 2


def train(setup_name, C, batch_size, workers, reuse_features, device):
    setup = config.get_setup(setup_name)
    DETECTOR_DIR.mkdir(parents=True, exist_ok=True)
    feature_file = DETECTOR_DIR / f"features_{setup_name}.npz"

    # 1. features (or the ones saved by an earlier run, with --reuse-features)
    if reuse_features and feature_file.exists():
        saved = np.load(feature_file)
        fake_features = saved["fake_features"]
        real_features = saved["real_features"]
        fake_ids = []
        for uid in saved["fake_ids"]:
            fake_ids.append(str(uid))  # numpy string -> plain Python string
        real_ids = []
        for uid in saved["real_ids"]:
            real_ids.append(str(uid))
        print(f"loaded features from {feature_file}")
    else:
        ap = AudioProcessor(device=device)
        detector = MelLogRegDetector().to(device)  # untrained: only its feature extraction is used here
        fake_features, fake_ids, real_features, real_ids = compute_features(
            setup, ap, detector, device, batch_size, workers)
        np.savez(feature_file, fake_features=fake_features, fake_ids=np.array(fake_ids),
                 real_features=real_features, real_ids=np.array(real_ids))

    # 2. split the utterances 80 / 20
    all_ids = sorted(real_ids)
    rng = np.random.RandomState(0)
    rng.shuffle(all_ids)
    n_test = len(all_ids) // 5
    test_ids = set(all_ids[:n_test])

    X_train, y_train, X_test, y_test = [], [], [], []
    for i in range(len(fake_ids)):
        if fake_ids[i] in test_ids:
            X_test.append(fake_features[i])
            y_test.append(1)
        else:
            X_train.append(fake_features[i])
            y_train.append(1)
    for i in range(len(real_ids)):
        if real_ids[i] in test_ids:
            X_test.append(real_features[i])
            y_test.append(0)
        else:
            X_train.append(real_features[i])
            y_train.append(0)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    # 3. standardise each mel band with the training mean / std, fit the logistic regression
    feature_mean = X_train.mean(axis=0)
    feature_std = X_train.std(axis=0) + 1e-8
    model = LogisticRegression(C=C, class_weight="balanced", max_iter=10000)
    model.fit((X_train - feature_mean) / feature_std, y_train)

    # 4. test accuracy (on utterances the detector never saw)
    test_logits = model.decision_function((X_test - feature_mean) / feature_std)
    fakes_called_fake = np.mean(test_logits[y_test == 1] > 0)
    reals_called_real = np.mean(test_logits[y_test == 0] <= 0)
    eer = equal_error_rate(y_test, test_logits)
    report = (
        f"{setup_name}: trained on {len(y_train)} clips ({int(y_train.sum())} fake), "
        f"tested on {len(y_test)} clips ({int(y_test.sum())} fake) from {n_test} held-out utterances\n"
        f"  C={C}\n"
        f"  test: fakes called fake {fakes_called_fake:.3f}, bona fide called bona fide {reals_called_real:.3f}, "
        f"balanced accuracy {(fakes_called_fake + reals_called_real) / 2:.3f}, EER {eer * 100:.2f} %"
    )
    print(report)

    out_path = detector_path(setup_name)
    torch.save({
        "feature_mean": torch.tensor(feature_mean, dtype=torch.float32),
        "feature_std": torch.tensor(feature_std, dtype=torch.float32),
        "weight": torch.tensor(model.coef_[0], dtype=torch.float32),
        "bias": torch.tensor(model.intercept_, dtype=torch.float32),
        "test_ids": sorted(test_ids),
        "report": report,
    }, out_path)
    (DETECTOR_DIR / f"mel_logreg_{setup_name}.txt").write_text(report + "\n")
    print(f"saved -> {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setup", choices=SETUP_NAMES, required=True)
    p.add_argument("--C", type=float, default=1.0, help="inverse regularisation strength of the logistic regression")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=16, help="DataLoader workers")
    p.add_argument("--reuse-features", action="store_true", help="refit from the features saved by an earlier run")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    train(args.setup, args.C, args.batch_size, args.workers, args.reuse_features, config.get_device(args.device))


if __name__ == "__main__":
    main()
