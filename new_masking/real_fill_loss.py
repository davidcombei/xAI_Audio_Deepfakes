"""The LMAC loss of ``losses.py``, with the masked-out bins filled from the bona fide clip."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import keep_masked_part, remove_masked_part


class RealFillLMACLoss(nn.Module):
    """Same three terms, learnable weights and initial weights as ``losses.LMACLoss``:

    * l_in  - keep the masked part (rest filled from the real clip) -> same decision
    * l_out - remove the masked part (filled from the real clip)   -> decision flips
    * L1    - mask mean (keep the mask small)

    fill="zero" gives back the original loss (used by sanity_check.py).
    """

    def __init__(self, audio_processor, detector, fill="real"):
        super().__init__()
        self.audio_processor = audio_processor
        self.detector = detector
        self.fill = fill
        self.w_raw = nn.Parameter(torch.tensor([3.5772736, 0.72915846, 3.914612]))

    @property
    def w(self):
        return F.softplus(self.w_raw)

    def loss_function(self, mask, x_fake, x_real, class_pred):
        """
        mask:       (batch, 1, 512, 248) U-Net output in [0, 1]
        x_fake:     (batch, 512, 248) complex STFT of the fake clips
        x_real:     (batch, 512, 248) complex STFT of their bona fide clips
        class_pred: (batch, 1) detector probability of "fake" on the fake clips
        """
        ap = self.audio_processor
        mask = mask.squeeze(1)  # (batch, 512, 248)
        relevant = keep_masked_part(mask, x_fake, x_real, self.fill)
        irrelevant = remove_masked_part(mask, x_fake, x_real, self.fill)

        # Run both inputs through wav2vec2 in one batch (the result is the same
        # as two separate passes: wav2vec2 normalises each clip on its own).
        both = torch.cat([relevant, irrelevant], dim=0)                     # (2*batch, 512, 248)
        features = ap.extract_features(ap.compute_invert_stft(both))        # (2*batch, frames, 1920)
        logits, _ = self.detector(features.mean(dim=1))                     # (2*batch, 1)
        batch_size = mask.shape[0]
        relevant_logits = logits[:batch_size]
        irrelevant_logits = logits[batch_size:]

        l_in = F.binary_cross_entropy_with_logits(relevant_logits, class_pred)
        l_out = F.binary_cross_entropy_with_logits(irrelevant_logits, 1 - class_pred)
        l1 = mask.abs().mean()

        losses = torch.stack([l_in, l_out, l1])
        total_loss = torch.sum(self.w * losses)
        return total_loss, losses, self.w
