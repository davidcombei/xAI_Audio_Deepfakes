from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LMACLoss(nn.Module):
    def __init__(self, audio_processor, detector, reg_w_tv: float = 0.0):
        super().__init__()
        self.audio_processor = audio_processor
        self.detector = detector
        self.w_raw = nn.Parameter(
            torch.tensor([3.5772736, 0.72915846, 3.914612], requires_grad=True)
        )

    @property
    def w(self) -> torch.Tensor:
        return F.softplus(self.w_raw)

    def loss_function(self, xhat, X_stft_power, X_stft_phase, class_pred):
        ap = self.audio_processor
        xhat = xhat.squeeze(1)
        Tmax = xhat.shape[1]

        power = X_stft_power[:, :Tmax, :]
        phase = torch.exp(1j * X_stft_phase[:, :Tmax, :])
        relevant = (xhat * power) * phase
        irrelevant = ((1 - xhat) * power) * phase

        rel_feats = torch.mean(ap.extract_features(ap.compute_invert_stft(relevant)), dim=1)
        irr_feats = torch.mean(ap.extract_features(ap.compute_invert_stft(irrelevant)), dim=1)

        rel_logits, _ = self.detector(rel_feats)
        irr_logits, _ = self.detector(irr_feats)

        l_in = F.binary_cross_entropy_with_logits(rel_logits, class_pred)
        l_out = F.binary_cross_entropy_with_logits(irr_logits, 1 - class_pred)
        reg_l1 = xhat.abs().mean()

        losses = torch.stack([l_in, l_out, reg_l1])
        total_loss = torch.sum(self.w * losses)

        return total_loss, losses, self.w
