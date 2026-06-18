"""Differentiable deepfake detectors (wav2vec2 features + logistic regression).

Two thin wrappers expose the frozen detector as an ``nn.Module`` so gradients can
flow back to the input - which is what the gradient-based explainers need:

* :class:`SpectralDetector` - input is ``(magnitude, phase)``; reconstructs the
  waveform via ISTFT, re-embeds it and scores it. Used by the mask predictor
  loss and the spectrogram-domain attributions.
* :class:`WaveformDetector` - input is a raw waveform. Used by waveform-domain
  attributions.

``build_detector`` wires a :class:`SpectralDetector` for a named setup.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config
from models import TorchLogReg


class SpectralDetector(nn.Module):


    def __init__(self, audio_processor, logreg):
        super().__init__()
        self.ap = audio_processor
        self.logreg = logreg


    def forward(self, magnitude, phase):
        spec = magnitude * torch.exp(1j * phase)
        audio = self.ap.compute_invert_stft(spec)
        feats = self.ap.extract_features(audio).mean(dim=1)
        logits, _ = self.logreg(feats)
        logits = logits.squeeze(1)
        return logits


class WaveformDetector(nn.Module):

    def __init__(self, audio_processor, logreg):
        super().__init__()
        self.ap = audio_processor
        self.logreg = logreg

    def forward(self, waveform):
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        feats = self.ap.extract_features(waveform).mean(dim=1)
        logits, _ = self.logreg(feats)
        return logits


def build_detector(
    setup,
    audio_processor,
    device: torch.device | str | None = None,
    negate: bool = False,
) -> SpectralDetector:
    if isinstance(setup, str):
        setup = config.get_setup(setup)
    device = config.get_device(device) if not isinstance(device, torch.device) else device
    logreg = TorchLogReg(setup.detector_ckpt).to(device)
    model = SpectralDetector(audio_processor, logreg).to(device)
    model.eval()
    return model
