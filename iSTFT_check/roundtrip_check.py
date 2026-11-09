"""Does the STFT -> mask -> iSTFT path change the audio, or the detector's answer?

No explanations here, only fixed masks. Every clip is masked in two ways:

  TF path (what the main pipeline does):
      waveform -> STFT -> multiply frame t of the spectrogram by mask[t] -> iSTFT -> detector
  waveform path (no STFT, no iSTFT):
      waveform * mask (one value per sample) -> detector

Masks:
  "identity"    all ones, nothing is masked: the TF path is a pure STFT -> iSTFT round trip
  "keep 3-5 s"  keep only the manipulated part of the time-swapped clips
  "keep 0-3 s"  keep only the real part

If the iSTFT is harmless, both paths give the same audio (very high SNR) and
the detector gives the same answer.

The main pipeline's STFT (AudioProcessor.compute_stft) also cuts the
spectrogram to 512 x 248: it drops the top frequency bin (the 8 kHz "Nyquist"
bin) and the last frame, and the iSTFT puts them back as zeros. The three
"round trip" rows test each cut on its own (no mask).

Run:     python iSTFT_check/roundtrip_check.py           (100-file list; add --all for all 5000 clips)
Output:  output/istft_check/roundtrip_<setup>_<subset|all>.csv   (one row per file and case)
"""

# The main ADDvisor modules (config, audio, detector, ...) live one folder up.
# Add that folder to Python's search path so we can import them.
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from common import (
    OUT_DIR,
    file_list,
    frames_to_samples,
    gt_time_profile,
    load_audio_processor,
    load_tf_detector,
    load_waveform_detector,
)
from data import AudioDataset


def main():
    args = parse_args()
    setup = config.get_setup(args.setup)
    device = config.get_device(args.device)

    audio_processor = load_audio_processor(device)
    tf_detector = load_tf_detector(setup, audio_processor, device)
    waveform_detector = load_waveform_detector(setup, audio_processor, device)

    paths = file_list(setup, args.use_all, args.limit)
    dataset = AudioDataset(paths, audio_processor, root=config.PROJECT_ROOT, device=device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    rows = []
    for waveform, audio_paths in tqdm(loader, desc="roundtrip", ascii=True):
        with torch.no_grad():
            rows = rows + check_batch(waveform, audio_paths, audio_processor, tf_detector, waveform_detector)

    table = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.use_all:
        out_file = OUT_DIR / f"roundtrip_{args.setup}_all.csv"
    else:
        out_file = OUT_DIR / f"roundtrip_{args.setup}_subset.csv"
    table.to_csv(out_file, index=False)
    print_summary(table)
    print(f"\nper-file rows -> {out_file}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setup", choices=config.SETUP_NAMES, default="time-swapped")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--all", dest="use_all", action="store_true",
                        help="every wav in the setup's data dir instead of its 100-file subset")
    parser.add_argument("--limit", type=int, default=None, help="only the first N files")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def check_batch(waveform, audio_paths, audio_processor, tf_detector, waveform_detector):
    """All cases for one batch of clips. Returns a list of table rows."""
    rows = []
    _, magnitude, phase = audio_processor.compute_stft(waveform)              # (batch, 512, 248) each
    clean_logit = waveform_detector(waveform)[:, 0]

    # Ground truth per frame: 1 for 3-5 s, 0 for 0-3 s.
    gt = []
    for path in audio_paths:
        gt.append(gt_time_profile(path))
    gt = torch.tensor(np.stack(gt), device=waveform.device)                   # (batch, 248)

    # 1. The three masks, through the TF path and through the waveform path.
    masks = [("identity", torch.ones_like(gt)), ("keep 3-5 s", gt), ("keep 0-3 s", 1 - gt)]
    for name, frame_mask in masks:
        masked_magnitude = magnitude * frame_mask.unsqueeze(1)     # the same value for every frequency of a frame
        tf_signal = audio_processor.compute_invert_stft(masked_magnitude * torch.exp(1j * phase))
        tf_logit = tf_detector(masked_magnitude, phase)

        wf_signal = frames_to_samples(frame_mask) * waveform
        wf_logit = waveform_detector(wf_signal)[:, 0]

        rows = rows + make_rows(name, audio_paths, wf_signal, tf_signal, wf_logit, tf_logit)

    # 2. STFT -> iSTFT round trips with no mask: no cut / only the Nyquist bin dropped / only the last frame dropped.
    round_trips = [
        ("round trip: no crop", False, False),
        ("round trip: Nyquist bin dropped", True, False),
        ("round trip: last frame dropped", False, True),
    ]
    for name, drop_nyquist, drop_last_frame in round_trips:
        signal = stft_then_istft(audio_processor, waveform, drop_nyquist, drop_last_frame)
        logit = waveform_detector(signal)[:, 0]
        rows = rows + make_rows(name, audio_paths, waveform, signal, clean_logit, logit)
    return rows


def make_rows(case, audio_paths, reference_signal, tf_signal, wf_logit, tf_logit):
    """One table row per clip, comparing the TF path with the waveform path."""
    rows = []
    for i in range(len(audio_paths)):
        p_tf = torch.sigmoid(tf_logit[i]).item()
        p_wf = torch.sigmoid(wf_logit[i]).item()
        rows.append({
            "file": audio_paths[i],
            "case": case,
            "snr_db": snr_db(reference_signal[i], tf_signal[i]),
            "p_tf": p_tf,                                         # P(fake), TF path
            "p_wf": p_wf,                                         # P(fake), waveform path
            "same_decision": float((p_tf > 0.5) == (p_wf > 0.5)),
        })
    return rows


def stft_then_istft(audio_processor, waveform, drop_nyquist, drop_last_frame):
    """STFT -> iSTFT with the main pipeline's settings, optionally zeroing the Nyquist bin or the last frame."""
    spectrogram = torch.stft(waveform, n_fft=audio_processor.n_fft, hop_length=audio_processor.hop_length,
                             win_length=audio_processor.win_length, return_complex=True)   # (batch, 513, 249)
    if drop_nyquist:
        spectrogram[:, -1, :] = 0          # the top frequency bin
    if drop_last_frame:
        spectrogram[:, :, -1] = 0          # the last frame
    return torch.istft(spectrogram, n_fft=audio_processor.n_fft, hop_length=audio_processor.hop_length,
                       win_length=audio_processor.win_length, length=waveform.shape[1])


def snr_db(reference, estimate):
    """Signal-to-noise ratio in dB: how close `estimate` is to `reference` (higher = closer).

    Capped at 120 dB: identical signals would give infinity.
    """
    signal_energy = (reference ** 2).sum().item()
    error_energy = ((reference - estimate) ** 2).sum().item()
    if error_energy == 0:
        return 120.0
    return min(120.0, 10 * np.log10((signal_energy + 1e-12) / error_energy))


def print_summary(table):
    """One line per case: median SNR, average P(fake) through both paths, share of same decisions."""
    summary = []
    for case in table["case"].unique():
        part = table[table["case"] == case]
        summary.append({
            "case": case,
            "SNR dB (median)": part["snr_db"].median(),
            "P(fake) TF": part["p_tf"].mean(),
            "P(fake) wf": part["p_wf"].mean(),
            "same decision": part["same_decision"].mean(),
        })
    print()
    print(pd.DataFrame(summary).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
