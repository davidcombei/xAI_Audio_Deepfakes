"""Audio front-end: loading, STFT/ISTFT and wav2vec2 feature extraction.

The :class:`AudioProcessor` bundles every signal-processing operation the
pipeline needs and keeps the STFT parameters consistent across training,
evaluation and visualisation. It pulls all defaults from :mod:`config` and loads
the (cached) wav2vec2 backbone lazily, so importing this module is cheap.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

import config
from models import load_wav2vec2, zero_mean_unit_var_norm


class AudioProcessor:
    def __init__(
        self,
        sampling_rate: int = config.SAMPLING_RATE,
        n_fft: int = config.N_FFT,
        hop_length: int = config.HOP_LENGTH,
        win_length: int = config.WIN_LENGTH,
        audio_length: int = config.AUDIO_LENGTH,
        device: torch.device | str | None = None,
    ):
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.audio_length = audio_length
        self.device = config.get_device(device) if not isinstance(device, torch.device) else device
        self._wav2vec2 = None  

    @property
    def wav2vec2(self):
        if self._wav2vec2 is None:
            self._wav2vec2 = load_wav2vec2(self.device)
        return self._wav2vec2

    def _fit_length(self, waveform: torch.Tensor, dim: int) -> torch.Tensor:
        length = int(self.audio_length * self.sampling_rate)
        current = waveform.shape[dim]
        if current < length:
            pad = [0, 0] * waveform.dim()
            pad[-(2 * dim + 1)] = length - current  
            waveform = F.pad(waveform, pad)
        else:
            waveform = waveform.narrow(dim, 0, length)
        return waveform

    def load_audio(self, audio_path: str | Path, target_sr: int | None = None):
        """Load a wav, downmix to mono, resample and pad/crop to a fixed length."""
        target_sr = target_sr or self.sampling_rate
        audio, sr = torchaudio.load(str(audio_path))
        if audio.ndim > 1:
            audio = audio.squeeze(0)
        if sr != target_sr:
            audio = T.Resample(orig_freq=sr, new_freq=target_sr)(audio)
        audio = self._fit_length(audio, dim=0)
        return audio, target_sr


    def extract_features(self, waveforms: torch.Tensor) -> torch.Tensor:

        single = waveforms.dim() == 1
        audio = waveforms.unsqueeze(0) if single else waveforms
        audio = zero_mean_unit_var_norm(audio).to(self.device)
        output = self.wav2vec2(audio, output_hidden_states=True)
        hidden = output.hidden_states[config.W2V2_HIDDEN_LAYER]
        return hidden.squeeze(0) if single else hidden

 
    def compute_stft(
        self, waveform: torch.Tensor, has_batch_dim: bool = True, logscale: bool = False
    ):

        if waveform.dim() == 1:
            waveform = self._fit_length(waveform, dim=0).to(self.device)
        elif waveform.dim() == 2:
            waveform = self._fit_length(waveform, dim=1).to(self.device)
        else:
            raise ValueError("waveform must be 1D (single) or 2D (batched waveforms)")

        X_stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True,
        )
  
        if has_batch_dim:
            X_stft = X_stft[:, :-1, :-1]
        else:
            X_stft = X_stft[:-1, :-1]

        magnitude = X_stft.abs()
        phase = X_stft.angle()
        if logscale:
            magnitude = torch.log1p(magnitude)
        return X_stft, magnitude, phase

    def compute_invert_stft(self, spectrogram: torch.Tensor) -> torch.Tensor:
  
        if not torch.is_complex(spectrogram):
            raise ValueError("ISTFT expects complex input!")

        if spectrogram.shape[1] == config.N_FREQ_BINS: 
            pad = torch.zeros(
                spectrogram.shape[0], 1, spectrogram.shape[2],
                dtype=spectrogram.dtype, device=spectrogram.device,
            )
            spectrogram = torch.cat([spectrogram, pad], dim=1)

        if spectrogram.shape[2] == config.N_TIME_FRAMES: 
            pad = torch.zeros(
                spectrogram.shape[0], spectrogram.shape[1], 1,
                dtype=spectrogram.dtype, device=spectrogram.device,
            )
            spectrogram = torch.cat([spectrogram, pad], dim=2)

        return torch.istft(
            spectrogram,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            length=self.audio_length * self.sampling_rate,
        )
