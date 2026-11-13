from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import config
from masking import keep_masked_part, remove_masked_part

EPS = 1e-10



def score_for_predicted_class(p: torch.Tensor) -> torch.Tensor:
    pred = (p > 0.5).float()
    return pred * p + (1 - pred) * (1 - p)


def fidelity(theta_out, predictions, threshold: float = 0.5) -> torch.Tensor:
    return ((predictions > threshold).long() == (theta_out > threshold).long()).float()


def faithfulness(predictions, predictions_masked) -> torch.Tensor:
    return ((predictions - predictions_masked) * torch.sign(predictions - 0.5)).squeeze(-1)


def average_drop(theta_out, predictions) -> torch.Tensor:
    pc = score_for_predicted_class(predictions.squeeze(-1))
    oc = score_for_predicted_class(theta_out.squeeze(-1))
    return (F.relu(pc - oc) / (pc + EPS)) * 100


def average_increase(theta_out, predictions) -> torch.Tensor:
    pc = score_for_predicted_class(predictions.squeeze(-1))
    oc = score_for_predicted_class(theta_out.squeeze(-1))
    return (oc > pc).float() * 100


def average_gain(theta_out, predictions) -> torch.Tensor:
    pc = score_for_predicted_class(predictions.squeeze(-1))
    oc = score_for_predicted_class(theta_out.squeeze(-1))
    return (F.relu(oc - pc) / (1 - pc + EPS)) * 100



def min_max_scale(x):
    x_min, x_max = x.min(), x.max()
    if x_max - x_min == 0:
        return np.zeros_like(x) if isinstance(x, np.ndarray) else torch.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


# Extra margin for "RMA (+50 Hz)": the swap's filter edges leak a little energy past the band,
# so this ground truth is the band widened by 50 Hz on each side (cut at 0 Hz and 8 kHz).
BOUNDARY_HZ = 50


def ground_truth_freq(audio_path, shape, margin_hz=0) -> torch.Tensor:
    """1 inside the swapped band (from the file name, e.g. '..._1000-2000.wav'), 0 elsewhere.

    margin_hz widens the band on both sides: band 1000-2000 with margin_hz=50 -> 950-2050 Hz.
    """
    n_freq, n_time = shape
    name = os.path.splitext(os.path.basename(str(audio_path)))[0]
    *_, band = name.split("_")
    f_low, f_high = (int(x) for x in band.split("-"))
    f_max = config.SAMPLING_RATE / 2
    f_low = max(f_low - margin_hz, 0)
    f_high = min(f_high + margin_hz, f_max)
    mask = torch.zeros((n_freq, n_time))
    mask[int(f_low / f_max * n_freq):int(f_high / f_max * n_freq)] = 1.0
    return mask


GROUND_TRUTH = {"freq": ground_truth_freq}



def detector_logits(detector, spectrogram):
    """Detector logit of a complex spectrogram (batch, 513, 313) -> (batch,)."""
    return detector(spectrogram.abs(), spectrogram.angle())


class FidelityEvaluator:
    """Share of clips where keeping only the masked part keeps the decision."""

    def __init__(self, detector, fill="real"):
        self.detector = detector
        self.fill = fill
        self.scores = []

    def evaluate_batch(self, mask, *, x_fake, x_real, audio_path=None):
        kept = keep_masked_part(mask, x_fake, x_real, self.fill)
        pred_orig = detector_logits(self.detector, x_fake) > 0
        pred_masked = detector_logits(self.detector, kept) > 0
        self.scores.extend((pred_orig == pred_masked).float().cpu().numpy())

    def results(self):
        return {"Fidelity": float(np.mean(self.scores))}


class FaithfulnessEvaluator:
    """Mean of p_fake(original) - p_fake(masked part removed)."""

    def __init__(self, detector, fill="real"):
        self.detector = detector
        self.fill = fill
        self.scores = []

    def evaluate_batch(self, mask, *, x_fake, x_real, audio_path=None):
        removed = remove_masked_part(mask, x_fake, x_real, self.fill)
        pred_orig = torch.sigmoid(detector_logits(self.detector, x_fake))
        pred_masked = torch.sigmoid(detector_logits(self.detector, removed))
        self.scores.extend((pred_orig - pred_masked).cpu().numpy())

    def results(self):
        return {"Faithfulness": float(np.mean(self.scores))}


class AverageDropIncreaseGain:

    def __init__(self, detector, fill="real"):
        self.detector = detector
        self.fill = fill
        self.clean, self.relevant = [], []

    def evaluate_batch(self, mask, *, x_fake, x_real, audio_path=None):
        kept = keep_masked_part(mask, x_fake, x_real, self.fill)
        self.clean.append(torch.sigmoid(detector_logits(self.detector, x_fake)).detach())
        self.relevant.append(torch.sigmoid(detector_logits(self.detector, kept)).detach())

    def results(self):
        clean = torch.cat(self.clean)
        relevant = torch.cat(self.relevant)
        return {
            "Average Drop": float(average_drop(relevant, clean).mean()),
            "Average Increase": float(average_increase(relevant, clean).mean()),
            "Average Gain": float(average_gain(relevant, clean).mean()),
        }


def relevance_mass_accuracy(attribution, true) -> float:
    """Relevance mass accuracy (Arras et al., 2022): share of total relevance inside the ground truth.

    Uses |attribution| so signed gradient attributions (input-x-gradient, gradient-shap) count
    by magnitude; for non-negative masks this is a no-op. The map is deliberately NOT min-max
    scaled: RMA is not shift-invariant, and scaling would zero out the smallest bin.
    Chance level = fraction of bins inside the ground truth.
    """
    a = np.abs(np.asarray(attribution, dtype=np.float64))
    total = a.sum()
    if total <= 0:
        return float("nan")
    return float(a[np.asarray(true, dtype=bool)].sum() / total)


class LocalizationEvaluator:

    def __init__(self, localization_kind: str):
        self.localization_kind = localization_kind
        self.get_true = GROUND_TRUTH[localization_kind]
        self.aucs = []
        self.rmas = []
        self.chances = []
        # frequency setups only: RMA with the band widened by BOUNDARY_HZ on each side
        self.rmas_boundary = []
        self.chances_boundary = []

    def evaluate_batch(self, pred, *, x_fake=None, x_real=None, audio_path):
        for i in range(pred.shape[0]):
            true = self.get_true(audio_path[i], pred[i].shape).cpu().numpy()
            pred_np = pred[i].cpu().numpy()
            pred_img = min_max_scale(pred_np)
            self.aucs.append(roc_auc_score(true.flatten(), pred_img.flatten()))
            self.rmas.append(relevance_mass_accuracy(pred_np, true))
            self.chances.append(float(true.mean()))
            if self.localization_kind == "freq":
                true_wide = ground_truth_freq(audio_path[i], pred[i].shape, BOUNDARY_HZ).cpu().numpy()
                self.rmas_boundary.append(relevance_mass_accuracy(pred_np, true_wide))
                self.chances_boundary.append(float(true_wide.mean()))

    def results(self):
        results = {
            "Localization": float(np.mean(self.aucs)),
            "Localization RMA": float(np.nanmean(self.rmas)),
            "Localization RMA (chance)": float(np.mean(self.chances)),
        }
        if self.rmas_boundary:
            results[f"Localization RMA +{BOUNDARY_HZ}Hz"] = float(np.nanmean(self.rmas_boundary))
            results[f"Localization RMA +{BOUNDARY_HZ}Hz (chance)"] = float(np.mean(self.chances_boundary))
        return results
