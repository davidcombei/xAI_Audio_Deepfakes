"""Shared helpers for the real-fill masking experiment.

The original pipeline removes the masked-out part of the spectrogram by
multiplying it with zero. Here the masked-out part is filled with the same
time-frequency bins of the *bona fide clip the fake was made from*:

    zero fill:  keep   = m * X_fake
                remove = (1 - m) * X_fake

    real fill:  keep   = m * X_fake + (1 - m) * X_real
                remove = (1 - m) * X_fake + m * X_real

X is the complex STFT (magnitude and phase). With a 0/1 mask, "keep" is the
fake inside the mask and the real clip outside it, and "remove" is the other
way round. Outside the manipulated region X_fake == X_real, so the mask value
there does not change either input.

The scripts import this file as ``common``; each script first adds the repo
root to ``sys.path`` so that ``config``, ``audio`` and ``models`` can be found.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

import config
from data import read_paths

EXPERIMENT_DIR = Path(__file__).resolve().parent
RUNS_DIR = EXPERIMENT_DIR / "runs"
RESULTS_DIR = EXPERIMENT_DIR / "results"

# The bona fide LJSpeech clips the freq-swap dataset was made from.
REAL_DIR = config.TRAIN_AUDIO_DIR

SETUP_NAMES = ["freq-swapped"]
FILLS = ["zero", "real"]


def utterance_id(path):
    """'LJSpeech_freq_swap/LJ001-0001.wav_vocoded_0-1000.wav' -> 'LJ001-0001'."""
    file_name = Path(path).name
    return file_name[:10]


def real_path(fake_path):
    """Path of the bona fide clip a fake clip was made from."""
    return REAL_DIR / (utterance_id(fake_path) + ".wav")


def list_files(setup, which="all"):
    """Repo-relative paths of the fake clips: every wav of the setup ("all"),
    or its 100-file evaluation list ("subset")."""
    if which == "subset":
        return read_paths(setup.filelist)
    paths = []
    for wav in sorted(setup.data_dir.glob("*.wav")):
        paths.append(str(wav.relative_to(config.PROJECT_ROOT)))
    return paths


class FakeAndRealDataset(Dataset):
    """Returns (fake waveform, bona fide waveform, fake path) for each fake clip.

    The waveforms stay on the CPU so DataLoader worker processes can load them.
    """

    def __init__(self, paths, audio_processor):
        self.paths = list(paths)
        self.ap = audio_processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        fake, _ = self.ap.load_audio(config.PROJECT_ROOT / path)  # (80000,)
        real, _ = self.ap.load_audio(real_path(path))             # (80000,)
        return fake, real, path


def collate_fake_and_real(batch):
    """Stack a list of dataset items into (fakes, reals, paths)."""
    fakes = []
    reals = []
    paths = []
    for fake, real, path in batch:
        fakes.append(fake)
        reals.append(real)
        paths.append(path)
    return torch.stack(fakes), torch.stack(reals), paths  # (batch, 80000), (batch, 80000), list


def one_thread_per_worker(worker_id):
    """Resampling runs on the CPU; one thread per DataLoader worker avoids
    oversubscribing the CPU when 8 GPU processes each start several workers."""
    torch.set_num_threads(1)


def complex_stft(ap, waveforms):
    """Complex STFT cropped to 512 x 248, exactly as the pipeline does it.

    waveforms: (batch, 80000) -> (batch, 512, 248) complex
    """
    stft, magnitude, phase = ap.compute_stft(waveforms, has_batch_dim=True)
    return stft


def keep_masked_part(mask, x_fake, x_real, fill):
    """LMAC "relevant" input: the fake inside the mask; zeros or the real clip outside."""
    if fill == "zero":
        return mask * x_fake
    return mask * x_fake + (1 - mask) * x_real


def remove_masked_part(mask, x_fake, x_real, fill):
    """LMAC "irrelevant" input: the fake outside the mask; zeros or the real clip inside."""
    if fill == "zero":
        return (1 - mask) * x_fake
    return (1 - mask) * x_fake + mask * x_real


def use_math_attention():
    # Same as train_mask.py: the math attention kernel keeps the backward
    # pass through wav2vec2 numerically stable.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def keep_only_needed_wav2vec2_layers(ap):
    """Drop the wav2vec2 layers above the one the detector reads.

    The checkpoint only has trained weights for the first 9 layers but its
    config declares 48, so the extra layers are random and never used. The
    detector reads hidden_states[9] (the input of layer 9), so keeping layers
    0..9 gives exactly the same features (checked: bit-identical) and makes
    every forward and backward pass about 4x faster.
    """
    model = ap.wav2vec2
    n_layers_to_keep = config.W2V2_HIDDEN_LAYER + 1
    if len(model.encoder.layers) > n_layers_to_keep:
        model.encoder.layers = model.encoder.layers[:n_layers_to_keep]


def detector_logits(ap, logreg, spectrogram):
    """Detector logit for each complex spectrogram: ISTFT -> wav2vec2 -> logistic regression.

    spectrogram: (batch, 512, 248) complex -> logits: (batch,)
    """
    waveforms = ap.compute_invert_stft(spectrogram)       # (batch, 80000)
    features = ap.extract_features(waveforms)              # (batch, frames, 1920)
    mean_features = features.mean(dim=1)                   # (batch, 1920)
    logits, probabilities = logreg(mean_features)          # (batch, 1)
    return logits.squeeze(1)
