

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
