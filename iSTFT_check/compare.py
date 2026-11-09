"""Evaluate explanations with and without the iSTFT.

For every clip and every method we compute two explanations:
  TF  the usual one: computed on the spectrogram, heard by the detector through the
      iSTFT (explainers.py of the main pipeline). We average it over frequency.
  wf  its waveform version: computed on the audio samples, no STFT and no iSTFT
      (waveform_explainers.py).

and score them with three numbers:
  auc           localization: does the explanation point at the manipulated 3-5 s?
                (ROC-AUC, 0.5 = chance; time-swapped setup only)
  fidelity      keep only what the mask keeps: does the detector give the same answer? (1 = yes)
  faithfulness  remove what the mask keeps: how much does P(fake) drop?
                (fidelity and faithfulness are only computed for masks:
                 optimized, trained, random)

Run:
  python iSTFT_check/compare.py                                   # time-swapped, 100-file list
  python iSTFT_check/compare.py --all --shard 0 --num-shards 8    # 1/8 of all 5000 clips (e.g. one per GPU)
  python iSTFT_check/compare.py --all --summarize-only            # merge the shards and summarize
  python iSTFT_check/compare.py --methods trained --waveform-mask-ckpt <FrameMaskNet .pth file>

Output (in output/istft_check/<setup>_<subset|all>/, or --out-dir):
  per_file*.csv           one row per file and method
  summary*.csv            averages per method
  summary_wellposed*.csv  the same, only on clips where "find 3-5 s" is a fair question
  plots/                  example figures
"""

# The main ADDvisor modules (config, audio, detector, ...) live one folder up.
# Add that folder to Python's search path so we can import them.
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse

import matplotlib

matplotlib.use("Agg")    # save figures to files (no screen needed)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from common import (
    OUT_DIR,
    SAMPLE_RATE,
    SWAP_ONSET_SEC,
    file_list,
    frame_times,
    frames_to_samples,
    gt_time_profile,
    load_audio_processor,
    load_tf_detector,
    load_waveform_detector,
    localization_auc,
    original_duration,
    use_stable_attention,
)
from data import AudioDataset
from explainers import build_explainer
from waveform_explainers import METHODS, explain_waveform
from waveform_model import load_frame_mask_net

DEFAULT_METHODS = ["optimized", "random", "saliency", "input-x-gradient", "gradient-shap"]
MASK_METHODS = ["optimized", "trained", "random"]      # explanations that are masks between 0 and 1


def main():
    args = parse_args()
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    elif args.use_all:
        out_dir = OUT_DIR / f"{args.setup}_all"
    else:
        out_dir = OUT_DIR / f"{args.setup}_subset"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize_only:
        table = read_all_results(out_dir)
    else:
        table = run_comparison(args, out_dir)
    print_summaries(table, args, out_dir)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setup", choices=["time-swapped", "freq-swapped", "realistic"], default="time-swapped")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=DEFAULT_METHODS,
                        help="'trained' needs --waveform-mask-ckpt")
    parser.add_argument("--waveform-mask-ckpt", default=None, help="FrameMaskNet checkpoint for method 'trained'")
    parser.add_argument("--all", dest="use_all", action="store_true",
                        help="every wav in the setup's data dir instead of its 100-file subset")
    parser.add_argument("--limit", type=int, default=None, help="only the first N files")
    parser.add_argument("--shard", type=int, default=0, help="which part of the files to run (0, 1, ...)")
    parser.add_argument("--num-shards", type=int, default=1, help="in how many parts the files are split")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=100, help="optimized waveform mask: Adam steps")
    parser.add_argument("--lr", type=float, default=0.1, help="optimized waveform mask: learning rate")
    parser.add_argument("--l1", type=float, default=1.0, help="optimized waveform mask: weight of the size term")
    parser.add_argument("--plots", type=int, default=4, help="number of example figures (shard 0 only)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None, help="default: output/istft_check/<setup>_<subset|all>")
    parser.add_argument("--summarize-only", action="store_true", help="merge existing per_file*.csv and summarize")
    return parser.parse_args()


def run_comparison(args, out_dir):
    """Explain and score every file of this shard. Saves and returns the per-file table."""
    setup = config.get_setup(args.setup)
    device = config.get_device(args.device)
    is_time_swap = setup.localization == "time"     # only then do we know which frames are manipulated

    # Fixed random seeds, so that "random" and "gradient-shap" give the same result every run.
    use_stable_attention()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # The same detector twice: spectrogram input (through the iSTFT) and waveform input.
    audio_processor = load_audio_processor(device)
    tf_detector = load_tf_detector(setup, audio_processor, device)
    waveform_detector = load_waveform_detector(setup, audio_processor, device)

    # TF explainers come from the main pipeline (explainers.py).
    # Both waveform masks ("optimized" and "trained") are compared with the
    # trained U-Net, which explainers.py calls "optimized".
    tf_explainers = {}
    for method in args.methods:
        if method == "trained":
            tf_explainers[method] = build_explainer(setup, "optimized", tf_detector, device=device)
        else:
            tf_explainers[method] = build_explainer(setup, method, tf_detector, device=device)

    trained_model = None
    if "trained" in args.methods:
        trained_model = load_frame_mask_net(args.waveform_mask_ckpt, device)

    # The files of this shard: every num_shards-th file, starting at file number `shard`.
    paths = file_list(setup, args.use_all, args.limit)
    paths = paths[args.shard::args.num_shards]
    dataset = AudioDataset(paths, audio_processor, root=config.PROJECT_ROOT, device=device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    rows = []
    n_plotted = 0
    for waveform, audio_paths in tqdm(loader, desc="compare", ascii=True):
        # The detector's answer on the untouched clips, through both paths.
        _, magnitude, phase = audio_processor.compute_stft(waveform)      # (batch, 512, 248) each
        with torch.no_grad():
            tf_logit = tf_detector(magnitude, phase)                       # (batch,)
            wf_logit = waveform_detector(waveform)[:, 0]                   # (batch,)

        profiles_for_plot = {}
        for method in args.methods:
            # 1. The two explanations.
            tf_explanation = tf_explainers[method](magnitude.detach(), phase, audio_paths)   # (batch, 512, 248)
            tf_explanation = tf_explanation.detach()
            tf_profile = tf_explanation.mean(dim=1)                                          # (batch, 248)
            wf_profile = explain_waveform(method, waveform, waveform_detector, trained_model,
                                          args.steps, args.lr, args.l1)                       # (batch, 248)

            # 2. Fidelity and faithfulness (masks only).
            if method in MASK_METHODS:
                fid_tf, faith_tf = tf_fidelity_faithfulness(tf_detector, magnitude, phase, tf_explanation, tf_logit)
                fid_wf, faith_wf = wf_fidelity_faithfulness(waveform_detector, waveform, wf_profile, wf_logit)

            # 3. One row per file.
            for i in range(len(audio_paths)):
                path = audio_paths[i]
                row = {
                    "file": path,
                    "method": method,
                    "orig_dur": float("nan"),
                    "p_tf": torch.sigmoid(tf_logit[i]).item(),     # detector's P(fake), TF path
                    "p_wf": torch.sigmoid(wf_logit[i]).item(),     # detector's P(fake), waveform path
                    "auc_tf": float("nan"),
                    "auc_wf": float("nan"),
                    "fid_tf": float("nan"),
                    "faith_tf": float("nan"),
                    "fid_wf": float("nan"),
                    "faith_wf": float("nan"),
                }
                if is_time_swap:
                    gt = gt_time_profile(path)
                    row["orig_dur"] = original_duration(path)
                    row["auc_tf"] = localization_auc(tf_profile[i].cpu().numpy(), gt)
                    row["auc_wf"] = localization_auc(wf_profile[i].cpu().numpy(), gt)
                if method in MASK_METHODS:
                    row["fid_tf"] = fid_tf[i].item()
                    row["faith_tf"] = faith_tf[i].item()
                    row["fid_wf"] = fid_wf[i].item()
                    row["faith_wf"] = faith_wf[i].item()
                rows.append(row)

            profiles_for_plot[method] = (tf_profile.cpu().numpy(), wf_profile.cpu().numpy())

        # 4. Example figures (first shard only).
        if args.shard == 0:
            for i in range(len(audio_paths)):
                if n_plotted < args.plots:
                    plot_one_file(audio_paths[i], waveform[i].cpu().numpy(), profiles_for_plot, i,
                                  is_time_swap, out_dir / "plots")
                    n_plotted = n_plotted + 1

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / f"per_file{file_suffix(args)}.csv", index=False)
    return table


def tf_fidelity_faithfulness(tf_detector, magnitude, phase, tf_mask, original_logit):
    """Fidelity and faithfulness of a TF mask (applied to the spectrogram, heard through the iSTFT)."""
    with torch.no_grad():
        kept_logit = tf_detector(magnitude * tf_mask, phase)
        removed_logit = tf_detector(magnitude * (1 - tf_mask), phase)
    fidelity = ((kept_logit > 0) == (original_logit > 0)).float()                  # same decision? 1 or 0
    faithfulness = torch.sigmoid(original_logit) - torch.sigmoid(removed_logit)   # drop in P(fake)
    return fidelity, faithfulness


def wf_fidelity_faithfulness(waveform_detector, waveform, frame_mask, original_logit):
    """Fidelity and faithfulness of a waveform mask (one value per frame, applied to the samples)."""
    sample_mask = frames_to_samples(frame_mask)
    with torch.no_grad():
        kept_logit = waveform_detector(sample_mask * waveform)[:, 0]
        removed_logit = waveform_detector((1 - sample_mask) * waveform)[:, 0]
    fidelity = ((kept_logit > 0) == (original_logit > 0)).float()
    faithfulness = torch.sigmoid(original_logit) - torch.sigmoid(removed_logit)
    return fidelity, faithfulness


def file_suffix(args):
    """Extra part of the output file names: the methods (if not the default ones) and the shard."""
    suffix = ""
    if args.methods != DEFAULT_METHODS:
        suffix = suffix + "_" + "+".join(args.methods)
    if args.num_shards > 1:
        suffix = suffix + f"_shard{args.shard}of{args.num_shards}"
    return suffix


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def read_all_results(out_dir):
    """Read and merge every per_file*.csv in the folder (e.g. the results of all shards)."""
    tables = []
    for csv_file in sorted(out_dir.glob("per_file*.csv")):
        tables.append(pd.read_csv(csv_file))
    return pd.concat(tables, ignore_index=True)


def summarize(table):
    """One row per method: number of files and the average of every score."""
    summary_rows = []
    for method in METHODS:
        part = table[table["method"] == method]
        if len(part) == 0:
            continue
        summary_rows.append({
            "method": method,
            "n": len(part),
            "auc_tf": part["auc_tf"].mean(),
            "auc_wf": part["auc_wf"].mean(),
            "fid_tf": part["fid_tf"].mean(),
            "faith_tf": part["faith_tf"].mean(),
            "fid_wf": part["fid_wf"].mean(),
            "faith_wf": part["faith_wf"].mean(),
        })
    return pd.DataFrame(summary_rows).set_index("method")


def print_summaries(table, args, out_dir):
    # Summary 1: all files.
    summaries = [("summary", table)]

    # Summary 2 (time-swapped only): the files where "find 3-5 s" is a fair question:
    #  - both detector paths call the clip fake (a mask explains the detector's decision), and
    #  - the original clip is at least 5 s long (shorter ones were padded with silence before the swap).
    if table["orig_dur"].notna().any():
        fair = table[(table["p_tf"] > 0.5) & (table["p_wf"] > 0.5) & (table["orig_dur"] >= 5)]
        summaries.append(("summary_wellposed", fair))

    # Runs of a non-default set of methods get their own summary files.
    suffix = ""
    if not args.summarize_only and args.methods != DEFAULT_METHODS:
        suffix = "_" + "+".join(args.methods)

    for name, part in summaries:
        summary = summarize(part)
        # One shard is only part of the files: save the summary of complete runs only.
        if args.num_shards == 1 or args.summarize_only:
            summary.to_csv(out_dir / f"{name}{suffix}.csv")
        print(f"\n== {name}: {part['file'].nunique()} files")
        print(summary.round(3).to_string())
    print(f"\nresults -> {out_dir}")


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def scale_0_to_1(values):
    """Rescale to [0, 1] so curves of different sizes can be drawn together."""
    values = np.asarray(values, dtype=np.float64)
    if values.max() == values.min():
        return np.zeros_like(values)
    return (values - values.min()) / (values.max() - values.min())


def plot_one_file(audio_path, waveform, profiles_for_plot, index, is_time_swap, plot_dir):
    """The waveform on top, then one row per method with its two time profiles (TF and waveform)."""
    methods = list(profiles_for_plot.keys())
    clip_seconds = len(waveform) / SAMPLE_RATE
    fig, axes = plt.subplots(len(methods) + 1, 1, figsize=(10, 1.8 * (len(methods) + 1)), sharex=True)

    # Top: the waveform.
    axes[0].plot(np.arange(len(waveform)) / SAMPLE_RATE, waveform, color="#52514e", linewidth=0.4)
    axes[0].set_title(Path(audio_path).name)
    axes[0].set_yticks([])

    # One row per method.
    for row in range(len(methods)):
        method = methods[row]
        tf_profiles, wf_profiles = profiles_for_plot[method]
        ax = axes[row + 1]
        ax.plot(frame_times(), scale_0_to_1(tf_profiles[index]), color="#2a78d6", label="TF (via iSTFT)")
        ax.plot(frame_times(), scale_0_to_1(wf_profiles[index]), color="#eb6834", label="waveform (no iSTFT)")
        ax.set_ylabel(method, fontsize=8)
        ax.set_ylim(-0.05, 1.05)

    # Shade the manipulated part, and add a legend.
    for ax in axes:
        if is_time_swap:
            ax.axvspan(SWAP_ONSET_SEC, clip_seconds, color="#d9d8d4", alpha=0.6, linewidth=0)
        ax.set_xlim(0, clip_seconds)
    axes[1].legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("time (s)")

    fig.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / f"{Path(audio_path).stem}.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
