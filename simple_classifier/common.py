"""Shared helpers for the simple-classifier experiment.

The data handling and the two ways of masking ("zero fill" = the original
ADDvisor pipeline, "real fill" = new_masking/) are reused from
``new_masking/common.py``. This file only adds the paths of this experiment
and the detector call.

The scripts import this file as ``common``; each script first adds the repo
root to ``sys.path`` so that ``config``, ``audio``, ``models`` and
``new_masking`` can be found.
"""

from __future__ import annotations

from pathlib import Path

from new_masking.common import (  # noqa: F401  (re-exported for the scripts)
    FILLS, SETUP_NAMES, FakeAndRealDataset, collate_fake_and_real, complex_stft,
    keep_masked_part, list_files, one_thread_per_worker, real_path, remove_masked_part,
    utterance_id,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
DETECTOR_DIR = EXPERIMENT_DIR / "detectors"
RUNS_DIR = EXPERIMENT_DIR / "runs"
RESULTS_DIR = EXPERIMENT_DIR / "results"


def detector_path(setup_name):
    """Checkpoint of the mel logistic regression trained for one setup."""
    return DETECTOR_DIR / f"mel_logreg_{setup_name}.pt"


def detector_logits(ap, detector, spectrogram):
    """Detector logit for each complex spectrogram: ISTFT -> mel logistic regression.

    spectrogram: (batch, 512, 248) complex -> logits: (batch,)
    """
    waveforms = ap.compute_invert_stft(spectrogram)  # (batch, 80000)
    return detector(waveforms)


def stft_round_trip(ap, waveforms):
    """What the detector sees of an unmasked clip in the pipeline: STFT (cropped
    to 512 x 248, as compute_stft does) -> ISTFT.

    waveforms: (batch, 80000) -> (batch, 80000)
    """
    return ap.compute_invert_stft(complex_stft(ap, waveforms))
