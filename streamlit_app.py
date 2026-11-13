"""Streamlit app that evaluates the trained U-Nets of three setups, one tab each.

    realistic            MLAAD fakes, real fill from M-AILABS
    freq-swapped         LJSpeech clips with one of 8 bands swapped
    freq-swapped-single  LJSpeech clips with the 1-2 kHz band swapped

For every setup the app picks 150 clips from the setup's file list:
    realistic            as many TTS systems and speakers as possible
    freq-swapped         the same number of clips for each of the 8 bands
    freq-swapped-single  random clips

The first start runs the U-Net and the detector on those clips (a few minutes
per setup) and saves the results to output/streamlit/<setup>.pt. Later starts
only load these files. Delete them to recompute (e.g. after changing a checkpoint).

Run from the repo root:
    python3 -m streamlit run streamlit_app.py --server.port 8531
"""

from __future__ import annotations

import os
import random
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.metrics import roc_auc_score

import config
from audio import AudioProcessor
from data import read_paths, real_path
from detector import build_detector
from explainers import build_mask_predictor
from masking import keep_masked_part, remove_masked_part
from metrics import GROUND_TRUTH, detector_logits, min_max_scale
from viz import plot_mask, plot_spectrogram

# Streamlit's file watcher crashes on torch.classes; this stops it from looking there.
if isinstance(torch.classes, types.ModuleType):
    torch.classes.__path__ = []

TAB_SETUPS = ["realistic", "freq-swapped", "freq-swapped-single"]
N_CLIPS = 150
SEED = 0
BATCH_SIZE = 4
RESULTS_DIR = config.PROJECT_ROOT / "output" / "streamlit"
MLAAD_FAKE_DIR = Path("/mnt/QNAP/comdav/DATA/DATA/MLAAD/fake")


# ---------------------------------------------------------------------------
# Information about one clip: its band (swaps) or its TTS system and speaker (MLAAD)
# ---------------------------------------------------------------------------

def band_of(path):
    """'LJSpeech_freq_swap/LJ001-0001.wav_vocoded_0-1000.wav' -> '0-1000'."""
    last_part = Path(path).name.split("_")[-1]   # '0-1000.wav'
    return last_part.replace(".wav", "")


def band_start(band):
    """'1000-2000' -> 1000 (used to sort the bands from low to high)."""
    return int(band.split("-")[0])


def speaker_of(path):
    """M-AILABS speaker of an MLAAD fake, read from the mailabs_real/ link.

    The link points to .../by_book/<gender>/<speaker>/<book>/wavs/<name>.wav,
    or .../by_book/mix/<book>/wavs/<name>.wav when the speaker is not known.
    """
    link = os.readlink(real_path(path, config.get_setup("realistic").real_dir))
    parts = Path(link).parts
    if parts[-4] == "mix":
        return "mix: " + parts[-3]
    return parts[-4]


def language_of(path):
    """M-AILABS language of an MLAAD fake, e.g. 'en_US'."""
    link = os.readlink(real_path(path, config.get_setup("realistic").real_dir))
    parts = Path(link).parts
    return parts[parts.index("m-ailabs") + 1]


def find_mlaad_systems(paths):
    """Return a dictionary: file name -> TTS system that made it ('unknown' if not found).

    MLAAD is stored as fake/<language>/<system>/<file name>.wav. The same file
    name can exist for several systems (same text, same source clip); then the
    file with the same size as ours is the one. Some files of mlaad_subset/ are
    not in this MLAAD folder (or have a different size); they get 'unknown'.
    """
    wanted_names = set()
    for path in paths:
        wanted_names.add(Path(path).name)

    # file name -> list of (system, full path)
    candidates = {}
    for language in sorted(os.listdir(MLAAD_FAKE_DIR)):
        language_dir = MLAAD_FAKE_DIR / language
        if not language_dir.is_dir():
            continue
        for system in sorted(os.listdir(language_dir)):
            system_dir = language_dir / system
            if not system_dir.is_dir():
                continue
            for file_name in os.listdir(system_dir):
                if file_name in wanted_names:
                    if file_name not in candidates:
                        candidates[file_name] = []
                    candidates[file_name].append((system, system_dir / file_name))

    systems = {}
    for path in paths:
        name = Path(path).name
        options = candidates.get(name, [])
        if len(options) > 1:
            our_size = os.path.getsize(config.PROJECT_ROOT / path)
            same_size = []
            for system, full_path in options:
                if os.path.getsize(full_path) == our_size:
                    same_size.append((system, full_path))
            options = same_size
        if len(options) == 1:
            systems[name] = options[0][0]
        else:
            systems[name] = "unknown"
    return systems


# ---------------------------------------------------------------------------
# Picking the 150 clips of each setup
# ---------------------------------------------------------------------------

def pick_realistic_clips(paths, n, seed):
    """Pick n MLAAD clips covering as many TTS systems, then speakers, as possible.

    One clip at a time we take the clip whose system has been picked the fewest
    times so far; among those, the one whose speaker has been picked the fewest
    times. Clips with an unknown system are only used when nothing else is left.
    """
    systems = find_mlaad_systems(paths)
    speakers = {}
    for path in paths:
        speakers[path] = speaker_of(path)

    remaining = list(paths)
    random.Random(seed).shuffle(remaining)   # random order breaks ties randomly

    times_system_picked = {}
    times_speaker_picked = {}
    clips = []
    while len(clips) < n and len(remaining) > 0:
        best_index = 0
        best_score = None
        for i, path in enumerate(remaining):
            system = systems[Path(path).name]
            if system == "unknown":
                system_score = 10 ** 9
            else:
                system_score = times_system_picked.get(system, 0)
            speaker_score = times_speaker_picked.get(speakers[path], 0)
            score = (system_score, speaker_score)
            if best_score is None or score < best_score:
                best_score = score
                best_index = i

        path = remaining.pop(best_index)
        system = systems[Path(path).name]
        speaker = speakers[path]
        times_system_picked[system] = times_system_picked.get(system, 0) + 1
        times_speaker_picked[speaker] = times_speaker_picked.get(speaker, 0) + 1
        clips.append({
            "path": path,
            "system": system,
            "speaker": speaker,
            "language": language_of(path),
        })
    return clips


def pick_freq_swap_clips(paths, n, seed):
    """Pick n clips with (almost) the same number of clips for every band."""
    paths_per_band = {}
    for path in paths:
        band = band_of(path)
        if band not in paths_per_band:
            paths_per_band[band] = []
        paths_per_band[band].append(path)

    rng = random.Random(seed)
    bands = sorted(paths_per_band, key=band_start)   # 0-1000, 1000-2000, ...
    for band in bands:
        rng.shuffle(paths_per_band[band])

    # take one clip of every band in turn until we have n clips
    clips = []
    round_number = 0
    while len(clips) < n:
        added = False
        for band in bands:
            if len(clips) < n and round_number < len(paths_per_band[band]):
                clips.append({"path": paths_per_band[band][round_number], "band": band})
                added = True
        if not added:
            break
        round_number += 1
    return clips


def pick_random_clips(paths, n, seed):
    """Pick n random clips."""
    chosen = random.Random(seed).sample(list(paths), min(n, len(paths)))
    clips = []
    for path in chosen:
        clips.append({"path": path, "band": band_of(path)})
    return clips


def pick_clips(setup_name):
    paths = read_paths(config.get_setup(setup_name).filelist)
    if setup_name == "realistic":
        return pick_realistic_clips(paths, N_CLIPS, SEED)
    if setup_name == "freq-swapped":
        return pick_freq_swap_clips(paths, N_CLIPS, SEED)
    return pick_random_clips(paths, N_CLIPS, SEED)


# ---------------------------------------------------------------------------
# Running the U-Net and the detector
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading wav2vec2 ...")
def load_audio_processor():
    return AudioProcessor(device=config.get_device())


def fill_of(setup):
    """Real fill when the setup has bona fide clips, else zero fill."""
    if setup.real_dir is not None:
        return "real"
    return "zero"


def load_fake_and_real(ap, setup, path):
    """(80000,) waveforms of the fake clip and of its bona fide clip (zeros if none)."""
    fake, _ = ap.load_audio(config.PROJECT_ROOT / path)
    if setup.real_dir is None:
        real = torch.zeros_like(fake)
    else:
        real, _ = ap.load_audio(real_path(path, setup.real_dir))
    return fake, real


def compute_results(setup_name, ap):
    """Run the U-Net and the detector on the setup's clips.

    Returns a dictionary:
        clips      list of dicts (path, band / system, speaker, ...)
        masks      (n_clips, 513, 313) float16 U-Net masks
        p_orig     p(fake) of the original clip
        p_kept     p(fake) when only the masked part is kept    (used for fidelity)
        p_removed  p(fake) when the masked part is removed      (used for faithfulness)
        auc        localization AUC of each mask (None for setups without ground truth)
    """
    setup = config.get_setup(setup_name)
    fill = fill_of(setup)
    detector = build_detector(setup, ap, device=ap.device)
    predict_mask = build_mask_predictor(setup, ap.device)
    clips = pick_clips(setup_name)

    masks = []
    p_orig = []
    p_kept = []
    p_removed = []
    aucs = []
    progress = st.progress(0.0, text=f"{setup_name}: running the U-Net and the detector ...")
    for start in range(0, len(clips), BATCH_SIZE):
        batch_clips = clips[start:start + BATCH_SIZE]
        fakes = []
        reals = []
        batch_paths = []
        for clip in batch_clips:
            fake, real = load_fake_and_real(ap, setup, clip["path"])
            fakes.append(fake)
            reals.append(real)
            batch_paths.append(clip["path"])

        with torch.no_grad():
            x_fake, magnitude, _ = ap.compute_stft(torch.stack(fakes).to(ap.device))   # (batch, 513, 313)
            x_real, _, _ = ap.compute_stft(torch.stack(reals).to(ap.device))           # (batch, 513, 313)
            mask = predict_mask(magnitude, batch_paths)                                 # (batch, 513, 313)
            kept = keep_masked_part(mask, x_fake, x_real, fill)
            removed = remove_masked_part(mask, x_fake, x_real, fill)
            prob_orig = torch.sigmoid(detector_logits(detector, x_fake))               # (batch,)
            prob_kept = torch.sigmoid(detector_logits(detector, kept))                 # (batch,)
            prob_removed = torch.sigmoid(detector_logits(detector, removed))           # (batch,)

        for i in range(len(batch_clips)):
            mask_np = mask[i].cpu().numpy()                                             # (513, 313)
            masks.append(torch.tensor(mask_np).half())
            p_orig.append(float(prob_orig[i]))
            p_kept.append(float(prob_kept[i]))
            p_removed.append(float(prob_removed[i]))
            if setup.localization is not None:
                true = GROUND_TRUTH[setup.localization](batch_paths[i], mask_np.shape).numpy()
                aucs.append(roc_auc_score(true.flatten(), min_max_scale(mask_np).flatten()))
        progress.progress(min(1.0, (start + BATCH_SIZE) / len(clips)),
                          text=f"{setup_name}: {min(start + BATCH_SIZE, len(clips))}/{len(clips)} clips")
    progress.empty()

    results = {
        "clips": clips,
        "masks": torch.stack(masks),
        "p_orig": np.array(p_orig),
        "p_kept": np.array(p_kept),
        "p_removed": np.array(p_removed),
        "auc": None,
    }
    if setup.localization is not None:
        results["auc"] = np.array(aucs)
    return results


@st.cache_resource(show_spinner=False)
def load_results(setup_name):
    """Results of one setup: loaded from output/streamlit/<setup>.pt, computed the first time."""
    results_file = RESULTS_DIR / f"{setup_name}.pt"
    if results_file.exists():
        return torch.load(results_file, weights_only=False)
    results = compute_results(setup_name, load_audio_processor())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(results, results_file)
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def summary_metrics(results, indices):
    """Fidelity, faithfulness (and localization AUC) over the clips in `indices`.

    Fidelity     share of clips where keeping only the masked part keeps the decision
    Faithfulness mean of p(fake | original) - p(fake | masked part removed)
    AUC          mean localization AUC of the mask against the swapped band
    Same definitions as metrics.py / evaluate.py.
    """
    same_decision = []
    drops = []
    aucs = []
    for i in indices:
        orig_is_fake = results["p_orig"][i] > 0.5
        kept_is_fake = results["p_kept"][i] > 0.5
        same_decision.append(float(orig_is_fake == kept_is_fake))
        drops.append(results["p_orig"][i] - results["p_removed"][i])
        if results["auc"] is not None:
            aucs.append(results["auc"][i])

    metrics = {
        "Clips": len(indices),
        "Fidelity": float(np.mean(same_decision)),
        "Faithfulness": float(np.mean(drops)),
    }
    if results["auc"] is not None:
        metrics["Localization AUC"] = float(np.mean(aucs))
    return metrics


def clip_label(clip):
    """Text shown in the clip picker, e.g. 'LJ001-0001  |  band 0-1000'."""
    name = Path(clip["path"]).name
    if "system" in clip:
        return f"{name}  |  {clip['system']}  |  {clip['speaker']} ({clip['language']})"
    return f"{name}  |  band {clip['band']}"


# ---------------------------------------------------------------------------
# One tab
# ---------------------------------------------------------------------------

def show_metrics(setup_name, results):
    all_clips = list(range(len(results["clips"])))
    metrics = summary_metrics(results, all_clips)
    columns = st.columns(len(metrics))
    for column, (name, value) in zip(columns, metrics.items()):
        if name == "Clips":
            column.metric(name, value)
        else:
            column.metric(name, f"{value:.3f}")

    if setup_name == "freq-swapped":
        st.markdown("**Per band**")
        rows = []
        bands = []
        for clip in results["clips"]:
            if clip["band"] not in bands:
                bands.append(clip["band"])
        for band in bands:
            indices = []
            for i, clip in enumerate(results["clips"]):
                if clip["band"] == band:
                    indices.append(i)
            row = {"band (Hz)": band}
            row.update(summary_metrics(results, indices))
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True)

    if setup_name == "realistic":
        systems = set()
        speakers = set()
        languages = set()
        for clip in results["clips"]:
            systems.add(clip["system"])
            speakers.add(clip["speaker"])
            languages.add(clip["language"])
        n_known_systems = len(systems - {"unknown"})
        st.caption(f"{n_known_systems} TTS systems, {len(speakers)} speakers, "
                   f"{len(languages)} languages")


def show_clip_table(results):
    rows = []
    for i, clip in enumerate(results["clips"]):
        row = {"file": Path(clip["path"]).name}
        if "system" in clip:
            row["system"] = clip["system"]
            row["speaker"] = clip["speaker"]
            row["language"] = clip["language"]
        else:
            row["band"] = clip["band"]
        row["p(fake) original"] = results["p_orig"][i]
        row["p(fake) relevant"] = results["p_kept"][i]
        row["p(fake) irrelevant"] = results["p_removed"][i]
        if results["auc"] is not None:
            row["AUC"] = results["auc"][i]
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, height=300)


def show_one_clip(setup_name, results):
    setup = config.get_setup(setup_name)
    labels = []
    for clip in results["clips"]:
        labels.append(clip_label(clip))
    chosen_label = st.selectbox("Clip (type to search a file, band, system or speaker)",
                                labels, key=f"clip_{setup_name}")
    index = labels.index(chosen_label)
    clip = results["clips"][index]
    mask = results["masks"][index].float()                                        # (513, 313)

    ap = load_audio_processor()
    fill = fill_of(setup)
    fake, real = load_fake_and_real(ap, setup, clip["path"])
    with torch.no_grad():
        x_fake, magnitude, _ = ap.compute_stft(fake)                              # (513, 313)
        x_real, _, _ = ap.compute_stft(real)                                      # (513, 313)
        mask_on_device = mask.to(x_fake.device)
        kept = keep_masked_part(mask_on_device, x_fake, x_real, fill)
        removed = remove_masked_part(mask_on_device, x_fake, x_real, fill)
        kept_audio = ap.compute_invert_stft(kept.unsqueeze(0))[0]                # (80000,)
        removed_audio = ap.compute_invert_stft(removed.unsqueeze(0))[0]          # (80000,)

    st.write(
        f"p(fake): original = **{results['p_orig'][index]:.3f}**  |  "
        f"relevant (mask kept) = **{results['p_kept'][index]:.3f}**  |  "
        f"irrelevant (mask removed) = **{results['p_removed'][index]:.3f}**"
    )
    if results["auc"] is not None:
        st.write(f"Localization AUC = **{results['auc'][index]:.3f}**")

    audio_columns = st.columns(4)
    audio_columns[0].markdown("**Fake (original)**")
    audio_columns[0].audio(fake.numpy(), sample_rate=config.SAMPLING_RATE)
    audio_columns[1].markdown("**Bona fide (fill source)**")
    audio_columns[1].audio(real.numpy(), sample_rate=config.SAMPLING_RATE)
    audio_columns[2].markdown("**Relevant (mask kept)**")
    audio_columns[2].audio(kept_audio.cpu().numpy(), sample_rate=config.SAMPLING_RATE)
    audio_columns[3].markdown("**Irrelevant (mask removed)**")
    audio_columns[3].audio(removed_audio.cpu().numpy(), sample_rate=config.SAMPLING_RATE)

    plot_columns = st.columns(3)
    figure = plot_spectrogram(magnitude.cpu(), title="fake spectrogram (log)", log=True)
    plot_columns[0].pyplot(figure)
    plt.close(figure)
    figure = plot_mask(mask, title="U-Net mask")
    plot_columns[1].pyplot(figure)
    plt.close(figure)
    if setup.localization is not None:
        true = GROUND_TRUTH[setup.localization](clip["path"], tuple(mask.shape))
        figure = plot_mask(true, title=f"ground truth (band {clip['band']} Hz)")
    else:
        figure = plot_mask(1 - mask, title="1 - mask")
    plot_columns[2].pyplot(figure)
    plt.close(figure)


def show_tab(setup_name):
    setup = config.get_setup(setup_name)
    st.caption(f"U-Net `{setup.mask_ckpt.relative_to(config.PROJECT_ROOT)}`  |  "
               f"detector `{setup.detector_ckpt.name}`  |  {fill_of(setup)} fill  |  "
               f"{N_CLIPS} clips from `{setup.filelist.name}`")
    results = load_results(setup_name)
    show_metrics(setup_name, results)
    st.markdown("**All clips**")
    show_clip_table(results)
    st.markdown("**One clip**")
    show_one_clip(setup_name, results)


st.set_page_config(layout="wide", page_title="ADDvisor U-Net evaluation")
st.title("ADDvisor - U-Net evaluation")

tabs = st.tabs(TAB_SETUPS)
for tab, setup_name in zip(tabs, TAB_SETUPS):
    with tab:
        show_tab(setup_name)
