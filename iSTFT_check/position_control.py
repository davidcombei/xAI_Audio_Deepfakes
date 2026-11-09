"""Does an explanation follow the manipulation, or the position it had during training?

In every time-swapped training clip the vocoded segment sits at 3-5 s. So a
mask predictor trained on them could "find" it just by its position. To test
this, we rebuild the swaps at other positions (same recipe as
vocoder.generate_time_swaps: take the clean clip and replace [start, end)
seconds with its vocoded version) and measure the localization AUC of:

  trained   the FrameMaskNet checkpoint (waveform, no iSTFT)
  unet      the trained TF U-Net, averaged over frequency
  saliency  waveform saliency (not trained, so it cannot know the usual position)

We only use clips whose original is at least 5 s long (so every position holds
speech), and the summary only counts clips the detector calls fake.

Run:
  python iSTFT_check/position_control.py \\
      --waveform-mask-ckpt runs/waveform_mask/time-swapped/FrameMaskNet_epoch_175_loss_1.1146.pth
Output:
  output/istft_check/position_control_<checkpoint name>.csv   (one row per file, position and explainer)
"""

# The main ADDvisor modules (config, audio, detector, ...) live one folder up.
# Add that folder to Python's search path so we can import them.
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse

import pandas as pd
import torch
from tqdm import tqdm

import config
from common import (
    OUT_DIR,
    SAMPLE_RATE,
    frame_times,
    load_audio_processor,
    load_waveform_detector,
    localization_auc,
    original_duration,
    use_stable_attention,
)
from data import read_paths
from explainers import build_mask_predictor
from waveform_explainers import explain_waveform
from waveform_model import load_frame_mask_net

POSITIONS = ["3-5", "2-4", "1-3", "0-2"]      # swap windows to test, in seconds ("start-end")


def main():
    args = parse_args()
    setup = config.get_setup("time-swapped")
    device = config.get_device(args.device)
    use_stable_attention()

    audio_processor = load_audio_processor(device)
    detector = load_waveform_detector(setup, audio_processor, device)
    frame_mask_net = load_frame_mask_net(args.waveform_mask_ckpt, device)
    unet = build_mask_predictor(setup, device)

    # Clips of the 100-file list whose original is at least 5 s long.
    names = []
    for path in read_paths(setup.filelist):
        name = Path(path).name
        if original_duration(name) >= 5:
            names.append(name)
    if args.limit is not None:
        names = names[:args.limit]

    rows = []
    for position in args.positions:
        start = float(position.split("-")[0])
        end = float(position.split("-")[1])
        # Ground truth: 1 for the frames inside the swap, 0 elsewhere.
        gt = ((frame_times() >= start) & (frame_times() < end)).astype(float)

        for first in tqdm(range(0, len(names), args.batch_size), desc=f"swap {position} s", ascii=True):
            batch_names = names[first:first + args.batch_size]
            clips = []
            for name in batch_names:
                clips.append(make_swapped_clip(audio_processor, name, start, end))
            waveform = torch.stack(clips).to(device)                       # (batch, 80000)

            with torch.no_grad():
                p_fake = torch.sigmoid(detector(waveform)[:, 0])            # (batch,)
                trained_profiles = frame_mask_net(waveform)                 # (batch, 248)
                _, magnitude, _ = audio_processor.compute_stft(waveform)
                unet_profiles = unet(magnitude).mean(dim=1)                 # (batch, 512, 248) -> (batch, 248)
            saliency_profiles = explain_waveform("saliency", waveform, detector)

            for i in range(len(batch_names)):
                for explainer, profiles in [("trained", trained_profiles),
                                            ("unet", unet_profiles),
                                            ("saliency", saliency_profiles)]:
                    rows.append({
                        "file": batch_names[i],
                        "swap": position,
                        "method": explainer,
                        "p_fake": p_fake[i].item(),
                        "auc": localization_auc(profiles[i].cpu().numpy(), gt),
                    })

    table = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"position_control_{Path(args.waveform_mask_ckpt).stem}.csv"
    table.to_csv(out_file, index=False)
    print_summary(table)
    print(f"\nper-file rows -> {out_file}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--waveform-mask-ckpt", required=True)
    parser.add_argument("--positions", nargs="+", default=POSITIONS, help="swap windows in seconds, e.g. 1-3")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def make_swapped_clip(audio_processor, name, start, end):
    """The clean clip with [start, end) seconds replaced by its vocoded version: (80000,)."""
    clean, _ = audio_processor.load_audio(config.TRAIN_AUDIO_DIR / name)
    vocoded, _ = audio_processor.load_audio(config.VOCODED_DIR / f"{name}_vocoded.wav")
    first_sample = int(start * SAMPLE_RATE)
    last_sample = int(end * SAMPLE_RATE)
    mixed = clean.clone()
    mixed[first_sample:last_sample] = vocoded[first_sample:last_sample]
    return mixed


def print_summary(table):
    """Average AUC per (position, explainer), over the clips the detector calls fake."""
    fake = table[table["p_fake"] > 0.5]
    summary = []
    for position in table["swap"].unique():
        for explainer in table["method"].unique():
            part = table[(table["swap"] == position) & (table["method"] == explainer)]
            part_fake = fake[(fake["swap"] == position) & (fake["method"] == explainer)]
            summary.append({
                "swap": position,
                "method": explainer,
                "n": len(part),
                "n_fake": len(part_fake),
                "auc": part_fake["auc"].mean(),
            })
    print(pd.DataFrame(summary).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
