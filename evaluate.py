
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import click
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from audio import AudioProcessor
from data import FakeAndRealDataset, collate_fake_and_real, read_paths
from detector import build_detector
from explainers import METHODS, build_explainer
from masking import FILLS, check_fill
from metrics import (
    GROUND_TRUTH,
    AverageDropIncreaseGain,
    FaithfulnessEvaluator,
    FidelityEvaluator,
    LocalizationEvaluator,
)

EXAMPLES_DIR = config.PROJECT_ROOT / "images_npy"


def pick_examples(paths, n, seed):
    """n random clips. The same seed gives the same clips for every method and fill."""
    if n <= 0:
        return []
    return random.Random(seed).sample(list(paths), min(n, len(paths)))


def clip_name(path):
    """'LJ004-0082.wav_vocoded_0-1000.wav' -> 'LJ004-0082_0-1000', '15YO_Capitan_s000013.wav' -> '15YO_Capitan_s000013'."""
    return Path(path).name.replace(".wav_vocoded_", "_").replace(".wav", "")


def save_example(out_dir, setup, method, path, waveform, mask):
    """Save one clip's audio, its explanation mask and (if known) its ground-truth mask.

    out_dir/<clip>.wav                the original wav, unchanged
    out_dir/<clip>_audio_16k.npy      (80000,) waveform the masks describe: 16 kHz, first 5 s
    out_dir/<clip>_<method>.npy       (513, 313) explanation mask, raw values
    out_dir/<clip>_ground_truth.npy   (513, 313) known manipulated region (controlled setups only)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    name = clip_name(path)
    shutil.copyfile(config.PROJECT_ROOT / path, out_dir / (name + ".wav"))
    np.save(out_dir / f"{name}_audio_16k.npy", waveform.cpu().numpy())
    mask = mask.detach().cpu().numpy()
    np.save(out_dir / f"{name}_{method}.npy", mask)
    if setup.localization is not None:
        true = GROUND_TRUTH[setup.localization](path, mask.shape).cpu().numpy()
        np.save(out_dir / f"{name}_ground_truth.npy", true)


def _cache(explainer, cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cached(magnitude, phase, audio_paths):
        paths = [cache_dir / (os.path.basename(p) + ".npy") for p in audio_paths]
        if all(p.exists() for p in paths):
            masks = [torch.tensor(np.load(p)) for p in paths]
            # masks cached with an older STFT have a different size: recompute them
            same_size = True
            for mask in masks:
                if tuple(mask.shape) != tuple(magnitude.shape[1:]):
                    same_size = False
            if same_size:
                return torch.stack(masks, dim=0).to(magnitude.device)
        expl = explainer(magnitude, phase, audio_paths)
        for mask, p in zip(expl.cpu().numpy(), paths):
            np.save(p, mask)
        return expl

    return cached


def method_label(method):
    """Name used in the printed results: the trained U-Net is the learned mask."""
    return "learned_mask" if method == "optimized" else method


def evaluate(setup, explanation_method, batch_size=4, device=None, cache=True, fill="real",
             n_examples=10, examples_seed=0, random_baseline=True):
    """Evaluate one explanation method and, as a baseline, random masks on the same clips.

    Returns (scores, example_files):
      scores        {label: {metric: value}}, e.g. {"learned_mask": {...}, "random": {...}}
      example_files the clips saved to images_npy/<setup>/ (as '<dir>/<clip>' prefixes)
    """
    setup = config.get_setup(setup) if isinstance(setup, str) else setup
    check_fill(setup, fill)
    device = config.get_device(device)
    ap = AudioProcessor(device=device)
    np.random.seed(examples_seed)  # random masks are reproducible (and cached after the first run)

    paths = read_paths(setup.filelist)
    examples = pick_examples(paths, n_examples, examples_seed)
    example_set = set(examples)
    examples_dir = EXAMPLES_DIR / setup.name
    real_dir = setup.real_dir if fill == "real" else None
    loader = DataLoader(
        FakeAndRealDataset(paths, ap, root=config.PROJECT_ROOT, real_dir=real_dir),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fake_and_real,
    )

    detector = build_detector(setup, ap, device=device)

    methods = [explanation_method]
    if random_baseline and explanation_method != "random":
        methods.append("random")

    explainers, evaluators = {}, {}
    for method in methods:
        explainer = build_explainer(setup, method, detector, device=device)
        if cache:
            explainer = _cache(explainer, config.PROJECT_ROOT / "output" / "explanations" / setup.name / method)
        explainers[method] = explainer
        evaluators[method] = [
            FidelityEvaluator(detector, fill),
            FaithfulnessEvaluator(detector, fill),
            AverageDropIncreaseGain(detector, fill),
        ]
        if setup.localization is not None:
            evaluators[method].append(LocalizationEvaluator(setup.localization))

    desc = f"{setup.name}/{'+'.join(methods)} ({fill} fill)"
    for fakes, reals, audio_paths in tqdm(loader, desc=desc, ascii=True):
        x_fake, magnitude, phase = ap.compute_stft(fakes.to(device))   # (batch, 513, 313)
        x_real, _, _ = ap.compute_stft(reals.to(device))               # (batch, 513, 313)
        for method in methods:
            pred = explainers[method](magnitude, phase, audio_paths)   # (batch, 513, 313)
            with torch.no_grad():
                for ev in evaluators[method]:
                    ev.evaluate_batch(pred, x_fake=x_fake, x_real=x_real, audio_path=audio_paths)
            for i, path in enumerate(audio_paths):
                if path in example_set:
                    save_example(examples_dir, setup, method, path, fakes[i], pred[i])

    scores = {}
    for method in methods:
        scores[method_label(method)] = {}
        for ev in evaluators[method]:
            scores[method_label(method)].update(ev.results())

    example_files = []
    if examples:
        for path in examples:
            example_files.append(str(examples_dir / clip_name(path)))
        with open(examples_dir / "files.txt", "w") as f:
            f.write("\n".join(examples) + "\n")
    return scores, example_files


@click.command()
@click.option("-s", "--setup", type=click.Choice(config.SETUP_NAMES), required=True)
@click.option("-e", "--explanation", "explanation_method", type=click.Choice(METHODS), required=True)
@click.option("-b", "--batch-size", default=4, show_default=True)
@click.option("--device", default=None, help="e.g. cuda, cuda:1, cpu")
@click.option("--cache/--no-cache", default=True, show_default=True)
@click.option("--fill", type=click.Choice(FILLS), default="real", show_default=True,
              help="real = masked-out bins filled from the bona fide clip; zero = original masking")
@click.option("-n", "--n-examples", default=10, show_default=True,
              help="random clips whose audio and masks are saved to images_npy/<setup>/ (0 = none)")
@click.option("--examples-seed", default=0, show_default=True,
              help="same seed = same clips for every method, so their masks can be compared; also seeds the random masks")
@click.option("--random-baseline/--no-random-baseline", default=True, show_default=True,
              help="also evaluate random masks on the same clips, as a baseline")
def main(setup, explanation_method, batch_size, device, cache, fill, n_examples, examples_seed,
         random_baseline):
    scores, example_files = evaluate(setup, explanation_method, batch_size, device, cache, fill,
                                     n_examples, examples_seed, random_baseline)
    for label, method_scores in scores.items():
        print(f"\n{label}:")
        for name, value in method_scores.items():
            print(f"  {name}: {value:.3f}")
    if example_files:
        suffixes = ["audio_16k", explanation_method]
        if "random" in scores and explanation_method != "random":
            suffixes.append("random")
        if config.get_setup(setup).localization is not None:
            suffixes.append("ground_truth")
        print(f"\n{len(example_files)} example clips saved (<clip>_{{{','.join(suffixes)}}}.npy and <clip>.wav):")
        for f in example_files:
            print(f"  {f}")


if __name__ == "__main__":
    main()
