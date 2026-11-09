"""Explanations computed directly on the waveform (no STFT, no iSTFT).

These are the waveform versions of the main pipeline's explainers.py. Every
method returns a time profile, one number per frame: (batch, 248).

Methods:
  optimized         fit a mask for each clip with the LMAC loss
  trained           a FrameMaskNet checkpoint from train_waveform_mask.py
  random            random values (a control: it should explain nothing)
  saliency          |gradient|, summed per frame
  input-x-gradient  waveform x gradient, summed per frame
  gradient-shap     GradientSHAP with silence as the baseline, summed per frame

"gradient" always means: how much the detector's logit changes when one audio
sample changes a tiny bit.
"""

# The main ADDvisor modules (config, audio, detector, ...) live one folder up.
# Add that folder to Python's search path so we can import them.
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from common import N_FRAMES, frames_to_samples, samples_to_frames

METHODS = ["optimized", "trained", "random", "saliency", "input-x-gradient", "gradient-shap"]


def explain_waveform(method, waveform, detector, trained_model=None, steps=100, lr=0.1, l1=1.0):
    """Explain a batch of waveforms (batch, 80000) with one method -> time profiles (batch, 248).

    detector       the waveform detector (common.load_waveform_detector)
    trained_model  a loaded FrameMaskNet, only needed for "trained"
    steps, lr, l1  settings of "optimized"
    """
    if method == "optimized":
        return optimized_mask(detector, waveform, steps, lr, l1)
    elif method == "trained":
        with torch.no_grad():
            return trained_model(waveform)
    elif method == "random":
        return random_profile(waveform.shape[0], waveform.device)
    elif method == "saliency":
        gradient = detector_gradient(detector, waveform)
        return samples_to_frames(gradient.abs())
    elif method == "input-x-gradient":
        gradient = detector_gradient(detector, waveform)
        return samples_to_frames(waveform * gradient)
    elif method == "gradient-shap":
        return gradient_shap(detector, waveform)
    else:
        raise ValueError(f"unknown method {method}, choose from {METHODS}")


def detector_gradient(detector, waveform):
    """Gradient of the detector's logit with respect to every audio sample: (batch, 80000)."""
    waveform = waveform.detach().clone()
    waveform.requires_grad_(True)
    with torch.enable_grad():
        logits = detector(waveform)                          # (batch, 1)
        # The clips in a batch don't influence each other, so the gradient of
        # the sum gives every clip the gradient of its own logit.
        gradient = torch.autograd.grad(logits.sum(), waveform)[0]
    return gradient.detach()


def gradient_shap(detector, waveform, n_samples=5):
    """GradientSHAP with silence (all zeros) as the baseline.

    Take n_samples random points on the way from silence to the clip
    (point = alpha * clip, alpha random between 0 and 1), compute the gradient
    at each point, multiply by the clip, and average.
    """
    total = torch.zeros_like(waveform)
    for _ in range(n_samples):
        alpha = torch.rand(waveform.shape[0], 1, device=waveform.device)    # one random number per clip
        point = alpha * waveform
        total = total + detector_gradient(detector, point) * waveform
    return samples_to_frames(total / n_samples)


def random_profile(batch_size, device):
    """A random 'explanation' (the control).

    The 248 frames are cut into segments of 5 to 30 frames, and every segment
    gets one random value between 0 and 1.
    """
    profiles = np.zeros((batch_size, N_FRAMES), dtype=np.float32)
    for clip in range(batch_size):
        start = 0
        while start < N_FRAMES:
            length = np.random.randint(5, 31)           # 5 to 30 frames
            profiles[clip, start:start + length] = np.random.uniform()
            start = start + length
    return torch.tensor(profiles, device=device)


def optimized_mask(detector, waveform, steps=100, lr=0.1, l1=1.0):
    """Find, separately for each clip, the time mask that best explains the detector's decision.

    Start from a mask of 0.5 everywhere and improve it with `steps` Adam steps
    on the LMAC loss (the same three terms as losses.LMACLoss):
      keep only the masked part:  the detector should still give its original answer
      keep only the rest:         the detector should give the opposite answer
      the mask should be small:   l1 * (average mask value)

    Returns the mask, one value between 0 and 1 per frame: (batch, 248).
    """
    # The detector's original answer, as a probability of "fake".
    with torch.no_grad():
        original_prob = torch.sigmoid(detector(waveform))          # (batch, 1)

    # We optimise free numbers and turn them into a mask with sigmoid
    # (sigmoid(0) = 0.5, so the mask starts at 0.5 everywhere).
    mask_numbers = torch.zeros(waveform.shape[0], N_FRAMES, device=waveform.device)
    mask_numbers.requires_grad_(True)
    optimizer = torch.optim.Adam([mask_numbers], lr=lr)

    with torch.enable_grad():
        for step in range(steps):
            frame_mask = torch.sigmoid(mask_numbers)               # (batch, 248)
            sample_mask = frames_to_samples(frame_mask)            # (batch, 80000)
            kept = sample_mask * waveform
            removed = (1 - sample_mask) * waveform

            loss_keep = F.binary_cross_entropy_with_logits(detector(kept), original_prob)
            loss_remove = F.binary_cross_entropy_with_logits(detector(removed), 1 - original_prob)
            loss_size = frame_mask.mean()
            loss = loss_keep + loss_remove + l1 * loss_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return torch.sigmoid(mask_numbers).detach()
