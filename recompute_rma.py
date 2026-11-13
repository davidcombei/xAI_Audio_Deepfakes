"""Recompute relevance mass accuracy (RMA) from the cached explanation masks.

No detector and no GPU: RMA only needs each clip's mask and its ground-truth band.
The masks are the ones evaluate.py cached in output/explanations/<setup>/<method>/<clip>.npy.

For every clip two RMA values are computed:
    RMA           relevance inside the swapped band, e.g. 1000-2000 Hz (same as evaluate.py)
    RMA +50Hz     relevance inside the band widened by 50 Hz on each side, e.g. 950-2050 Hz

    python recompute_rma.py -s freq-swapped -e saliency          one setup / method
    python recompute_rma.py --all                                every frequency setup and method

Each run saves logs/evaluation/rma/<setup>__<method>.txt; --summary prints all of them as one table.
"""

from __future__ import annotations

import os
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

import config
from data import read_paths
from explainers import METHODS
from metrics import BOUNDARY_HZ, ground_truth_freq, relevance_mass_accuracy

CACHE_DIR = config.PROJECT_ROOT / "output" / "explanations"
OUT_DIR = config.PROJECT_ROOT / "logs" / "evaluation" / "rma"
FREQ_SETUPS = ["freq-swapped-single", "freq-swapped"]


def recompute(setup_name, method):
    """Mean RMA (band and band +50 Hz) and the chance levels over every clip of the setup."""
    setup = config.get_setup(setup_name)
    paths = read_paths(setup.filelist)
    mask_dir = CACHE_DIR / setup_name / method

    rmas, rmas_wide = [], []
    chances, chances_wide = [], []
    missing = 0
    for path in tqdm(paths, desc=f"{setup_name}/{method}", ascii=True):
        mask_file = mask_dir / (os.path.basename(path) + ".npy")
        if not mask_file.exists():
            missing += 1
            continue
        mask = np.load(mask_file)                                        # (513, 313)
        true = ground_truth_freq(path, mask.shape).numpy()               # (513, 313), 0/1
        true_wide = ground_truth_freq(path, mask.shape, BOUNDARY_HZ).numpy()
        rmas.append(relevance_mass_accuracy(mask, true))
        rmas_wide.append(relevance_mass_accuracy(mask, true_wide))
        chances.append(float(true.mean()))
        chances_wide.append(float(true_wide.mean()))

    return {
        "clips": len(rmas),
        "missing masks": missing,
        "RMA": float(np.nanmean(rmas)),
        "RMA chance": float(np.mean(chances)),
        f"RMA +{BOUNDARY_HZ}Hz": float(np.nanmean(rmas_wide)),
        f"RMA +{BOUNDARY_HZ}Hz chance": float(np.mean(chances_wide)),
    }


def save(setup_name, method, results):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{setup_name}__{method}.txt", "w") as f:
        for name, value in results.items():
            f.write(f"{name}: {value}\n")


def print_summary():
    columns = None
    print(f"{'setup':<20}  {'method':<17}", end="")
    for txt in sorted(OUT_DIR.glob("*.txt")):
        setup_name, method = txt.stem.split("__")
        values = {}
        for line in txt.read_text().splitlines():
            name, value = line.rsplit(": ", 1)
            values[name] = float(value)
        if columns is None:
            columns = list(values)
            for name in columns:
                print(f"  {name:>15}", end="")
            print()
        print(f"{setup_name:<20}  {method:<17}", end="")
        for name in columns:
            if name in ("clips", "missing masks"):
                print(f"  {int(values[name]):>15d}", end="")
            else:
                print(f"  {values[name]:>15.3f}", end="")
        print()


@click.command()
@click.option("-s", "--setup", "setup_name", type=click.Choice(FREQ_SETUPS))
@click.option("-e", "--explanation", "method", type=click.Choice(METHODS))
@click.option("--all", "run_all", is_flag=True, help="every frequency setup and every method")
@click.option("--summary", is_flag=True, help="only print the saved results")
def main(setup_name, method, run_all, summary):
    if not summary:
        jobs = []
        if run_all:
            for s in FREQ_SETUPS:
                for m in METHODS:
                    jobs.append((s, m))
        else:
            if setup_name is None or method is None:
                raise click.UsageError("give -s and -e, or --all, or --summary")
            jobs.append((setup_name, method))
        for s, m in jobs:
            save(s, m, recompute(s, m))
    print_summary()


if __name__ == "__main__":
    main()
