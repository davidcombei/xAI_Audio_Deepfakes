"""The simple detector: logistic regression on the log-mel spectrogram averaged over time.

    waveform, 5 s at 16 kHz                     (80000 samples)
      -> mel spectrogram                        (80 mel bands x 249 frames; Hann window,
                                                 n_fft 1024, hop 322, like the pipeline's STFT)
      -> log(mel + 1e-6)
      -> mean over time                         (80 numbers per clip)
      -> standardise with the training-set mean and std of each band
      -> logistic regression                    (80 weights + 1 bias) -> logit, > 0 means fake

It replaces "wav2vec2 + logistic regression" and keeps the same interface:
waveforms in, one logit per clip out. So the ADDvisor pipeline around it
(mask on the STFT -> ISTFT -> detector) stays exactly the same.

Because the model is linear in the log-mel values, the logit splits exactly
into one contribution per mel band and frame (see ``contributions``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio

import config

LOG_OFFSET = 1e-6  # log(mel + LOG_OFFSET): keeps log() finite on silent (all-zero) frames
EVIDENCE_THRESHOLD = 1e-3  # see evidence_mask()


class MelLogRegDetector(nn.Module):

    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLING_RATE,
            n_fft=config.N_FFT,
            win_length=config.WIN_LENGTH,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
        )
        # Filled in by train_detector.py (saved in the checkpoint).
        self.register_buffer("feature_mean", torch.zeros(config.N_MELS))
        self.register_buffer("feature_std", torch.ones(config.N_MELS))
        self.linear = nn.Linear(config.N_MELS, 1)
        for param in self.linear.parameters():
            param.requires_grad = False  # the detector is frozen; only the U-Net is trained

        if checkpoint_path is not None:
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
            self.set_weights(checkpoint["feature_mean"], checkpoint["feature_std"],
                             checkpoint["weight"], checkpoint["bias"])
        self.eval()

    def set_weights(self, feature_mean, feature_std, weight, bias):
        """Copy the standardisation and logistic-regression numbers into the module."""
        with torch.no_grad():
            self.feature_mean.copy_(torch.as_tensor(feature_mean, dtype=torch.float32))
            self.feature_std.copy_(torch.as_tensor(feature_std, dtype=torch.float32))
            self.linear.weight.copy_(torch.as_tensor(weight, dtype=torch.float32).view(1, config.N_MELS))
            self.linear.bias.copy_(torch.as_tensor(bias, dtype=torch.float32).view(1))

    def log_mel(self, waveforms):
        """waveforms: (batch, 80000) -> log-mel spectrogram: (batch, 80, 249)"""
        return torch.log(self.mel(waveforms) + LOG_OFFSET)

    def features(self, waveforms):
        """waveforms: (batch, 80000) -> log-mel averaged over time: (batch, 80)"""
        return self.log_mel(waveforms).mean(dim=2)

    def forward(self, waveforms):
        """waveforms: (batch, 80000) -> logits: (batch,)   (> 0 means fake)"""
        standardised = (self.features(waveforms) - self.feature_mean) / self.feature_std  # (batch, 80)
        logits = self.linear(standardised)                                               # (batch, 1)
        return logits.squeeze(1)

    def band_weights(self):
        """How much one unit of mean log-mel in each band moves the logit: (80,)"""
        return self.linear.weight.view(config.N_MELS) / self.feature_std

    def contributions(self, fake_waveforms, real_waveforms):
        """Exact split of  logit(fake) - logit(real)  over mel bands and frames.

        logit = bias + sum_k  w_k / std_k * (mean_t logmel[k, t] - mean_k), so

            logit(fake) - logit(real) = sum over k, t of
                w_k / std_k * (logmel_fake[k, t] - logmel_real[k, t]) / n_frames

        Each term is the contribution of mel band k in frame t; > 0 pushes
        towards "fake". Returns (batch, 80, 249); summing it gives the logit difference.
        """
        difference = self.log_mel(fake_waveforms) - self.log_mel(real_waveforms)  # (batch, 80, 249)
        n_frames = difference.shape[2]
        weights = self.band_weights().view(1, config.N_MELS, 1)                    # (1, 80, 1)
        return weights * difference / n_frames

    def evidence_mask(self, fake_waveforms, real_waveforms):
        """The detector's own explanation, as a mask on the pipeline's 512 x 248 STFT grid.

        1. contributions(): (batch, 80 mel bands, 249 frames)
        2. share each band's contribution among the STFT bins of its mel filter,
           in proportion to the filter weights (so the total stays the same)
        3. mask = 1 where a bin pushes towards "fake" by more than
           EVIDENCE_THRESHOLD x the clip's strongest bin, else 0

        Why a threshold: outside the swapped region the fake and the bona fide
        clip differ only by rounding noise, which gives tiny contributions of
        either sign in about half of all bins. On 24 sample clips per setup,
        the bins above 0.1 % of the strongest one hold ~96 % of the positive
        evidence (15 % of the bins for freq swaps).

        The mel spectrogram uses the same hop as the pipeline's STFT, so frame t
        is the same 20 ms in both; the last frame is dropped like compute_stft does.
        Returns (batch, 512, 248) with values 0 / 1.
        """
        contributions = self.contributions(fake_waveforms, real_waveforms)  # (batch, 80, 249)
        filters = self.mel.mel_scale.fb                                      # (513, 80): STFT bin x mel band
        filters = filters[:config.N_FREQ_BINS, :]                            # (512, 80): drop the Nyquist bin
        filter_sums = filters.sum(dim=0, keepdim=True)                       # (1, 80)
        filter_shares = filters / torch.clamp(filter_sums, min=1e-12)        # each column sums to 1

        per_bin = torch.matmul(filter_shares, contributions)                 # (batch, 512, 249)
        per_bin = per_bin[:, :, :config.N_TIME_FRAMES]                       # (batch, 512, 248)
        largest = per_bin.amax(dim=(1, 2), keepdim=True)                     # (batch, 1, 1)
        mask = per_bin > EVIDENCE_THRESHOLD * torch.clamp(largest, min=1e-12)
        return mask.float()
