"""The trainable waveform mask predictor (FrameMaskNet) and its training loss.

They are the waveform versions of the main pipeline's models.UNet and losses.LMACLoss:

  UNet          spectrogram -> one mask value per (frequency, frame) -> iSTFT -> detector
  FrameMaskNet  waveform    -> one mask value per frame -> multiply the waveform -> detector

Nothing here uses the STFT or the iSTFT.

Note: the layer names (filterbank, e1, e2, ...) are stored inside the saved
checkpoints. Renaming them would make those checkpoints impossible to load.
"""

# The main ADDvisor modules (config, audio, detector, ...) live one folder up.
# Add that folder to Python's search path so we can import them.
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import HOP_LENGTH, N_FRAMES, WIN_LENGTH, frames_to_samples
from models import zero_mean_unit_var_norm


class ConvBlock1d(nn.Module):
    """Two 1-D convolutions, each followed by batch norm and LeakyReLU (1-D copy of models.ConvBlock)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FrameMaskNet(nn.Module):
    """Predict a mask with one value per frame, straight from the waveform.

    Input:  waveform (batch, 80000)
    Output: mask     (batch, 248), values between 0 and 1

    1. A learned filterbank cuts the audio into the same 248 frames as the STFT
       (644-sample windows, one every 322 samples). It plays the role of the STFT.
    2. A small 1-D U-Net looks at the sequence of frames
       (248 -> 124 -> 62 frames and back up, with skip connections).
    3. A sigmoid gives one value between 0 and 1 per frame.
    """

    def __init__(self, n_filters=256, width=256):
        super().__init__()
        # 1. learned filterbank
        self.filterbank = nn.Conv1d(1, n_filters, kernel_size=WIN_LENGTH, stride=HOP_LENGTH,
                                    padding=WIN_LENGTH // 2, bias=False)
        self.norm = nn.BatchNorm1d(n_filters)

        # 2. U-Net. Encoder ("e"):
        self.e1 = ConvBlock1d(n_filters, width)                 # 248 frames
        self.e2 = ConvBlock1d(width, width, stride=2)           # 124 frames
        self.e3 = ConvBlock1d(width, 2 * width, stride=2)       # 62 frames
        # Middle: dilated convolutions look at a wide stretch of time.
        self.bottleneck = nn.Sequential(
            nn.Conv1d(2 * width, 2 * width, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(2 * width),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(2 * width, 2 * width, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(2 * width),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Decoder ("up" = upsample, "d" = decode):
        self.up3 = nn.ConvTranspose1d(2 * width, width, kernel_size=2, stride=2)   # 62 -> 124 frames
        self.d3 = ConvBlock1d(2 * width, width)
        self.up2 = nn.ConvTranspose1d(width, width, kernel_size=2, stride=2)       # 124 -> 248 frames
        self.d2 = ConvBlock1d(2 * width, width)

        # 3. one value per frame, squashed between 0 and 1
        self.mask_head = nn.Sequential(nn.Conv1d(width, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, waveform):
        # Normalise every clip like the detector does (zero mean, unit variance).
        audio = zero_mean_unit_var_norm(waveform)          # (batch, 80000)
        audio = audio.unsqueeze(1)                         # (batch, 1, 80000)

        # 1. Filterbank: gives 249 frames; drop the last one, like the main pipeline's STFT does.
        frames = self.filterbank(audio)                    # (batch, 256, 249)
        frames = frames[:, :, :N_FRAMES]                   # (batch, 256, 248)
        frames = torch.log1p(frames.abs())                 # log-magnitude, like a spectrogram
        frames = self.norm(frames)

        # 2. U-Net
        enc1 = self.e1(frames)                             # (batch, 256, 248)
        enc2 = self.e2(enc1)                               # (batch, 256, 124)
        enc3 = self.e3(enc2)                               # (batch, 512, 62)
        middle = self.bottleneck(enc3)                     # (batch, 512, 62)
        dec3 = self.up3(middle)                            # (batch, 256, 124)
        dec3 = self.d3(torch.cat([dec3, enc2], dim=1))     # skip connection -> (batch, 256, 124)
        dec2 = self.up2(dec3)                              # (batch, 256, 248)
        dec2 = self.d2(torch.cat([dec2, enc1], dim=1))     # skip connection -> (batch, 256, 248)

        # 3. Mask
        mask = self.mask_head(dec2)                        # (batch, 1, 248)
        return mask.squeeze(1)                             # (batch, 248)


def load_frame_mask_net(checkpoint_path, device):
    """Load a FrameMaskNet checkpoint saved by train_waveform_mask.py, ready to use."""
    model = FrameMaskNet().to(device)
    saved_weights = torch.load(str(checkpoint_path), map_location=device)
    # Checkpoints saved from a multi-GPU wrapper have "module." in front of every name.
    weights = {}
    for name in saved_weights:
        weights[name.replace("module.", "")] = saved_weights[name]
    model.load_state_dict(weights)
    model.eval()
    return model


class WaveformLMACLoss(nn.Module):
    """The training loss of the main pipeline (losses.LMACLoss), with the mask applied to the waveform.

    Three terms:
      loss_keep    keep only the masked part: the detector should still give its original answer
      loss_remove  keep only the rest: the detector should give the opposite answer
      loss_size    the mask should be as small as possible

    Like in losses.LMACLoss, the three weights are learned too. They are stored
    as raw numbers (w_raw) and made positive with softplus.
    """

    def __init__(self, audio_processor, detector_head):
        super().__init__()
        self.audio_processor = audio_processor     # turns audio into wav2vec2 features
        self.detector = detector_head              # logistic regression on the averaged features
        self.w_raw = nn.Parameter(torch.tensor([3.5772736, 0.72915846, 3.914612]))   # same start as LMACLoss

    def weights(self):
        """The three (positive) weights: [keep, remove, size]."""
        return F.softplus(self.w_raw)

    def detector_logit(self, waveform):
        """The detector's logit for a batch of waveforms: (batch, 1)."""
        features = self.audio_processor.extract_features(waveform)    # (batch, n_steps, 1920)
        average_features = features.mean(dim=1)                       # (batch, 1920)
        logit, _ = self.detector(average_features)
        return logit

    def loss_function(self, frame_mask, waveform, original_prob):
        """frame_mask (batch, 248), waveform (batch, 80000), original_prob (batch, 1) = the detector's P(fake).

        Returns: total loss, the three terms, the three weights.
        """
        sample_mask = frames_to_samples(frame_mask)         # (batch, 80000)
        kept = sample_mask * waveform                       # the part the mask keeps
        removed = (1 - sample_mask) * waveform              # everything else

        loss_keep = F.binary_cross_entropy_with_logits(self.detector_logit(kept), original_prob)
        loss_remove = F.binary_cross_entropy_with_logits(self.detector_logit(removed), 1 - original_prob)
        loss_size = frame_mask.abs().mean()

        terms = torch.stack([loss_keep, loss_remove, loss_size])
        weights = self.weights()
        total = (weights * terms).sum()
        return total, terms, weights
