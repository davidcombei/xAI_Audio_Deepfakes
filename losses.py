from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from masking import keep_masked_part, remove_masked_part

# Fixed weights of the three loss terms. Change them here by hand.
# (These are the values the old learnable weights started from.)
W_IN = 3    # keep the masked part   -> same decision
W_OUT = 1  # remove the masked part -> decision flips
W_L1 = 1.5   # mask mean (keep the mask small)


class LMACLoss(nn.Module):
    """LMAC loss with fixed term weights (W_IN, W_OUT, W_L1 above):

    * l_in  - keep the masked part (rest zero / filled from the real clip) -> same decision
    * L_out - remove the masked part (zero / filled from the real clip)    -> decision flips
    * L1    - mask mean (keep the mask small)

    fill="real" fills the masked-out bins from the bona fide clip (see masking.py);
    fill="zero" is the original loss (x_real is then ignored).
    """

    def __init__(self, audio_processor, detector, fill: str = "real", reg_w_tv: float = 0.0):
        super().__init__()
        self.audio_processor = audio_processor
        self.detector = detector
        self.fill = fill
        # a buffer, not a parameter: moves to the GPU with .to(device) but is never trained
        self.register_buffer("w", torch.tensor([W_IN, W_OUT, W_L1]))

    def loss_function(self, mask, x_fake, x_real, class_pred):
        """
        mask:       (batch, 1, 513, 313) U-Net output in [0, 1]
        x_fake:     (batch, 513, 313) complex STFT of the fake clips
        x_real:     (batch, 513, 313) complex STFT of their bona fide clips (unused for zero fill)
        class_pred: (batch, 1) detector probability of "fake" on the fake clips
        """
        ap = self.audio_processor
        mask = mask.squeeze(1)  # (batch, 513, 313)
        relevant = keep_masked_part(mask, x_fake, x_real, self.fill)
        irrelevant = remove_masked_part(mask, x_fake, x_real, self.fill)

        # Run both inputs through wav2vec2 in one batch (the result is the same
        # as two separate passes: wav2vec2 normalises each clip on its own).
        both = torch.cat([relevant, irrelevant], dim=0)                     # (2*batch, 513, 313)
        features = ap.extract_features(ap.compute_invert_stft(both))        # (2*batch, frames, 1920)
        logits, _ = self.detector(features.mean(dim=1))                     # (2*batch, 1)
        batch_size = mask.shape[0]
        rel_logits = logits[:batch_size]
        irr_logits = logits[batch_size:]

        l_in = F.binary_cross_entropy_with_logits(rel_logits, class_pred)
        l_out = F.binary_cross_entropy_with_logits(irr_logits, 1 - class_pred)
        reg_l1 = mask.abs().mean()

        losses = torch.stack([l_in, l_out, reg_l1])
        total_loss = torch.sum(self.w * losses)

        return total_loss, losses, self.w
