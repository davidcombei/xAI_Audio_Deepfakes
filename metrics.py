from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import config

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


def ground_truth_freq(audio_path, shape) -> torch.Tensor:
    n_freq, n_time = shape
    name = os.path.splitext(os.path.basename(str(audio_path)))[0]
    *_, band = name.split("_")
    f_low, f_high = (int(x) for x in band.split("-"))
    f_max = config.SAMPLING_RATE / 2
    mask = torch.zeros((n_freq, n_time))
    mask[int(f_low / f_max * n_freq):int(f_high / f_max * n_freq)] = 1.0
    return mask


def ground_truth_time(audio_path, shape) -> torch.Tensor:
    n_freq, n_time = shape
    mask = torch.zeros((n_freq, n_time))
    mask[:, int(3 / 5 * n_time):] = 1.0
    return mask


GROUND_TRUTH = {"freq": ground_truth_freq, "time": ground_truth_time}



class FidelityEvaluator:
    def __init__(self, detector):
        self.detector = detector
        self.scores = []

    def evaluate_batch(self, mask, *, magnitude, phase, audio_path=None):
        pred_orig = self.detector(magnitude, phase) > 0
        pred_masked = self.detector(magnitude * mask, phase) > 0
        self.scores.extend((pred_orig == pred_masked).float().cpu().numpy())

    def results(self):
        return {"Fidelity": float(np.mean(self.scores))}


class FaithfulnessEvaluator:
    def __init__(self, detector):
        self.detector = detector
        self.scores = []

    def evaluate_batch(self, mask, *, magnitude, phase, audio_path=None):
        pred_orig = torch.sigmoid(self.detector(magnitude, phase))
        pred_masked = torch.sigmoid(self.detector(magnitude * (1 - mask), phase))
        self.scores.extend((pred_orig - pred_masked).cpu().numpy())

    def results(self):
        return {"Faithfulness": float(np.mean(self.scores))}


class AverageDropIncreaseGain:

    def __init__(self, detector):
        self.detector = detector
        self.clean, self.relevant = [], []

    def evaluate_batch(self, mask, *, magnitude, phase, audio_path=None):
        self.clean.append(torch.sigmoid(self.detector(magnitude, phase)).detach())
        self.relevant.append(torch.sigmoid(self.detector(magnitude * mask, phase)).detach())

    def results(self):
        clean = torch.cat(self.clean)
        relevant = torch.cat(self.relevant)
        return {
            "Average Drop": float(average_drop(relevant, clean).mean()),
            "Average Increase": float(average_increase(relevant, clean).mean()),
            "Average Gain": float(average_gain(relevant, clean).mean()),
        }


class LocalizationEvaluator:

    def __init__(self, localization_kind: str):
        self.get_true = GROUND_TRUTH[localization_kind]
        self.aucs = []

    def evaluate_batch(self, pred, *, magnitude, phase, audio_path):
        for i in range(pred.shape[0]):
            true = self.get_true(audio_path[i], pred[i].shape).cpu().numpy()
            pred_img = min_max_scale(pred[i].cpu().numpy())
            self.aucs.append(roc_auc_score(true.flatten(), pred_img.flatten()))

    def results(self):
        return {"Localization": float(np.mean(self.aucs))}
