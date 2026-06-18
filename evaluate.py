
from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import click
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from audio import AudioProcessor
from data import AudioDataset, collate_features, read_paths
from detector import build_detector
from explainers import METHODS, build_explainer
from metrics import (
    AverageDropIncreaseGain,
    FaithfulnessEvaluator,
    FidelityEvaluator,
    LocalizationEvaluator,
)


def _cache(explainer, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cached(magnitude, phase, audio_paths):
        paths = [cache_dir / (os.path.basename(p) + ".npy") for p in audio_paths]
        if all(p.exists() for p in paths):
            masks = [torch.tensor(np.load(p)) for p in paths]
            return torch.stack(masks, dim=0).to(magnitude.device)
        expl = explainer(magnitude, phase)
        for mask, p in zip(expl.cpu().numpy(), paths):
            np.save(p, mask)
        return expl

    return cached


def evaluate(setup, explanation_method, batch_size=4, device=None, cache=True):
    setup = config.get_setup(setup) if isinstance(setup, str) else setup
    device = config.get_device(device)
    ap = AudioProcessor(device=device)

    paths = read_paths(setup.filelist)
    loader = DataLoader(
        AudioDataset(paths, ap, root=config.PROJECT_ROOT, device=device),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_features, audio_processor=ap),
    )

    detector = build_detector(setup, ap, device=device)
    explainer = build_explainer(setup, explanation_method, detector, device=device)
    if cache:
        explainer = _cache(
            explainer, config.PROJECT_ROOT / "output" / "explanations" / setup.name / explanation_method
        )

    evaluators = [
        FidelityEvaluator(detector),
        FaithfulnessEvaluator(detector),
        AverageDropIncreaseGain(detector),
    ]
    if setup.localization is not None:
        evaluators.append(LocalizationEvaluator(setup.localization))

    for _, magnitude, phase, audio_paths in tqdm(loader, desc=f"{setup.name}/{explanation_method}", ascii=True):
        pred = explainer(magnitude, phase, audio_paths) if cache else explainer(magnitude, phase)
        for ev in evaluators:
            ev.evaluate_batch(pred, magnitude=magnitude, phase=phase, audio_path=audio_paths)

    scores = {}
    for ev in evaluators:
        scores.update(ev.results())
    return scores


@click.command()
@click.option("-s", "--setup", type=click.Choice(config.SETUP_NAMES), required=True)
@click.option("-e", "--explanation", "explanation_method", type=click.Choice(METHODS), required=True)
@click.option("-b", "--batch-size", default=4, show_default=True)
@click.option("--device", default=None, help="e.g. cuda, cuda:1, cpu")
@click.option("--cache/--no-cache", default=True, show_default=True)
def main(setup, explanation_method, batch_size, device, cache):
    scores = evaluate(setup, explanation_method, batch_size, device, cache)
    print()
    for name, value in scores.items():
        print(f"{name}: {value:.2f}")


if __name__ == "__main__":
    main()
