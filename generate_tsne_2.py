from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.manifold import TSNE

import config

sns.set(style="white", context="poster")


def _paths(setup_key, real_path):
    d = config.masked_features_dir(setup_key)
    return {
        "Real": real_path,
        "Fake": d / f"features_original_{setup_key}.npy",
        "Fake x Mask": d / f"features_relevant_{setup_key}.npy",
        "Fake x (1 - Mask)": d / f"features_irrelevant_{setup_key}.npy",
    }


PANELS = {
    "Realistic": _paths("realistic", config.FEATURES_DIR / "real_features.npy"),
    "Controlled: Freq.": _paths("freq", config.masked_features_dir("freq") / "features_real_freq.npy"),
}


def show_tsne(ax, paths, n=2000, to_show_legend=False):
    xs, labels = [], []
    for label, path in paths.items():
        if not path.exists():
            continue
        data = np.load(path)
        xs.append(data)
        labels.extend([label] * len(data))
    X = np.vstack(xs)
    labels = np.array(labels)
    if len(X) > n:
        idx = np.random.choice(len(X), size=n, replace=False)
        X, labels = X[idx], labels[idx]

    embedded = TSNE(n_components=2).fit_transform(X)
    df = pd.DataFrame(embedded, columns=["TSNE1", "TSNE2"])
    df["Label"] = labels
    sns.scatterplot(data=df, x="TSNE1", y="TSNE2", hue="Label", legend=to_show_legend, ax=ax)
    ax.set(xticklabels=[], yticklabels=[], xlabel="", ylabel="")


fig, axs = plt.subplots(1, len(PANELS), figsize=(6, 6))
for ax, (title, paths) in zip(axs, PANELS.items()):
    show_tsne(ax, paths, to_show_legend=(title == "Realistic"))
    ax.set_title(title)
sns.move_legend(axs[0], "upper right", bbox_to_anchor=(1.3, 1))
st.pyplot(fig)
