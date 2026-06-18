from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

import config

PALETTE = {
    "Real": "green",
    "Fake": "red",
    "Fake * Mask": "purple",
    "Fake * (1-Mask)": "orange",
}


def run_tsne(arrays: dict[str, np.ndarray], out_path, max_per_class=1000, seed=42):
    """Plot a t-SNE scatter of ``{label: features}`` to ``out_path``."""
    rng = random.Random(seed)
    xs, ys = [], []
    for label, data in arrays.items():
        if data is None or len(data) == 0:
            continue
        if max_per_class and len(data) > max_per_class:
            idx = rng.sample(range(len(data)), max_per_class)
            data = data[idx]
        xs.append(data)
        ys.extend([label] * len(data))
    X = np.vstack(xs)

    print(f"running t-SNE on {X.shape[0]} points ...")
    embedded = TSNE(n_components=2, perplexity=30, random_state=seed, max_iter=10000).fit_transform(X)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedded[:, 0], y=embedded[:, 1], hue=ys, palette=PALETTE, s=15, alpha=0.7)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(title="type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved -> {out_path}")


def _load(path):
    path = Path(path) if path else None
    return np.load(path) if path and path.exists() else None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setup", choices=config.SETUP_NAMES, required=True)
    p.add_argument("--real", default=None)
    p.add_argument("--fake", default=None)
    p.add_argument("--relevant", default=None)
    p.add_argument("--irrelevant", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--max-per-class", type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    setup = config.get_setup(args.setup)
    key = config.setup_key(setup)
    feat_dir = config.masked_features_dir(setup)

    arrays = {
        "Real": _load(args.real or (config.FEATURES_DIR / "real_features.npy")),
        "Fake": _load(args.fake or (feat_dir / f"features_original_{key}.npy")),
        "Fake * Mask": _load(args.relevant or (feat_dir / f"features_relevant_{key}.npy")),
        "Fake * (1-Mask)": _load(args.irrelevant or (feat_dir / f"features_irrelevant_{key}.npy")),
    }
    out = args.out or (config.VIZ_DIR / "tsne" / f"tsne_{key}.png")
    run_tsne(arrays, out, max_per_class=args.max_per_class)


if __name__ == "__main__":
    main()
