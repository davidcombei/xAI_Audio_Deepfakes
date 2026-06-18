from __future__ import annotations

import numpy as np
import torch

import config
from models import load_mask_predictor

METHODS = ("optimized", "random", "saliency", "input-x-gradient", "gradient-shap")


def build_explainer(setup, method: str, detector, device=None):
    if isinstance(setup, str):
        setup = config.get_setup(setup)
    device = config.get_device(device) if not isinstance(device, torch.device) else device

    if method == "optimized":
        decoder = load_mask_predictor(setup.mask_ckpt, device)

        def explainer(magnitude, phase):
            with torch.no_grad():
                return decoder(magnitude.unsqueeze(1)).squeeze(1)

        return explainer

    if method == "random":
        try:
            from skimage.segmentation import felzenszwalb
        except ImportError as e:
            raise ImportError(
                "needs scikit-image (`pip install scikit-image`)."
            ) from e

        def _segment(mag):
            segments = felzenszwalb(mag, scale=15, sigma=0.5)
            out = np.zeros_like(mag)
            for seg_id in np.unique(segments):
                out[segments == seg_id] = np.random.uniform(0, 1)
            return out

        def explainer(magnitude, phase):
            expl = np.stack([_segment(m) for m in magnitude.cpu().numpy()], axis=0)
            return torch.tensor(expl).to(device)

        return explainer

    if method in ("saliency", "input-x-gradient", "gradient-shap"):
        from captum.attr import GradientShap, InputXGradient, Saliency

        factories = {
            "saliency": Saliency,
            "input-x-gradient": InputXGradient,
            "gradient-shap": GradientShap,
        }
        attributor = factories[method](detector)

        def explainer(magnitude, phase):
            magnitude = magnitude.requires_grad_(True)
            kwargs = {}
            if method == "gradient-shap":
                kwargs["baselines"] = torch.zeros_like(magnitude)
            attr = attributor.attribute(
                magnitude, additional_forward_args=(phase,), **kwargs
            )
            return attr.detach()

        return explainer

    raise ValueError(f"unknown explanation method {method!r}. Choose from {METHODS}.")
