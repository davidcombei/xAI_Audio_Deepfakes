"""The LMAC loss of ``new_masking/real_fill_loss.py``, with the mel logistic regression as detector."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import keep_masked_part, remove_masked_part


class MelLMACLoss(nn.Module):
    """Same three terms, learnable weights and initial weights as ``losses.LMACLoss``:

    * l_in  - keep the masked part     -> same decision
    * l_out - remove the masked part   -> decision flips
    * L1    - mask mean (keep the mask small)

    fill="zero": the removed part is set to zero (original ADDvisor).
    fill="real": the removed part is taken from the bona fide clip (new_masking/).
    """

    def __init__(self, audio_processor, detector, fill="zero"):
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
        x_real:     (batch, 512, 248) complex STFT of their bona fide clips (unused with zero fill)
        class_pred: (batch,) detector probability of "fake" on the fake clips
        """
        ap = self.audio_processor
        mask = mask.squeeze(1)  # (batch, 512, 248)
        relevant = keep_masked_part(mask, x_fake, x_real, self.fill)
        irrelevant = remove_masked_part(mask, x_fake, x_real, self.fill)

        relevant_logits = self.detector(ap.compute_invert_stft(relevant))      # (batch,)
        irrelevant_logits = self.detector(ap.compute_invert_stft(irrelevant))  # (batch,)

        l_in = F.binary_cross_entropy_with_logits(relevant_logits, class_pred)
        l_out = F.binary_cross_entropy_with_logits(irrelevant_logits, 1 - class_pred)
        l1 = mask.abs().mean()

        losses = torch.stack([l_in, l_out, l1])
        total_loss = torch.sum(self.w * losses)
        return total_loss, losses, self.w
