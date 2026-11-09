"""Save the audio and the explanation masks of a few clips as .npy files.

Run from anywhere:
    python3 images_npy/export_npys.py --device cuda:1

For every clip this writes one folder with:
    audio_original.wav       the wav file, unchanged
    audio_16k.npy            the waveform the masks describe: (80000,), 16 kHz, first 5 s
    mask_ground_truth.npy    the known manipulated region, 0 or 1
    mask_decoder.npy         output of the trained U-Net decoder, 0 to 1
    mask_saliency.npy        Captum Saliency, raw values (not scaled to 0-1)
    mask_input_x_gradient.npy  Captum InputXGradient, raw values (can be negative)

All masks are (513, 313) = (frequency bins, time frames), except for the time
clip's decoder. That checkpoint is from before the 2026-10-29 STFT change: it has
the old decoder and only works on the old (512, 248) STFT. The time clip therefore
gets these instead of mask_decoder.npy:
    mask_decoder_512x248.npy         the decoder's real output
    mask_decoder_resized_513x313.npy the same mask resized, to compare with the others
and one extra file to compare with the real output:
    mask_ground_truth_512x248.npy
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))  # so we can import the repo's root modules

import config
from audio import AudioProcessor
from detector import build_detector
from explainers import build_explainer
from metrics import GROUND_TRUTH
from models import UNet

# Which clips to export. Change these lists to export other clips.
# All eight bands of this utterance are in metadata/freq_visualization_metadata.txt.
FREQ_FILES = [
    "LJSpeech_freq_swap/LJ004-0082.wav_vocoded_0-1000.wav",
    "LJSpeech_freq_swap/LJ004-0082.wav_vocoded_3000-4000.wav",
    "LJSpeech_freq_swap/LJ004-0082.wav_vocoded_6000-7000.wav",
]
TIME_FILES = [
    "LJSpeech_time_swap/LJ041-0009.wav",
]

# The STFT and the decoder layers the old time checkpoint was trained with.
OLD_N_FFT = 1024
OLD_HOP = 322
OLD_WIN = 644


class OldDecoderUNet(UNet):
    """The U-Net as it was before 2026-10-29: only the four upsampling layers differ."""

    def __init__(self):
        super().__init__()
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=(2, 2))
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2))
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=(2, 1), stride=(2, 1))
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=(2, 1), stride=(2, 1))


def old_decoder_mask(waveform, checkpoint_path, device):
    """Run the old time checkpoint on its old STFT: (1, 80000) -> (512, 248)."""
    model = OldDecoderUNet().to(device)
    state = torch.load(str(checkpoint_path), map_location=device)
    clean_state = {}
    for key in state:
        clean_state[key.replace("module.", "")] = state[key]  # saved from DataParallel
    model.load_state_dict(clean_state)
    model.eval()

    spectrogram = torch.stft(
        waveform,
        n_fft=OLD_N_FFT,
        hop_length=OLD_HOP,
        win_length=OLD_WIN,
        return_complex=True,
    )                                                 # (1, 513, 249)
    magnitude = spectrogram[:, :-1, :-1].abs()        # (1, 512, 248), the old crop
    with torch.no_grad():
        mask = model(magnitude.unsqueeze(1))          # (1, 1, 512, 248)
    return mask


def save_mask(folder, name, mask):
    """Save one (freq, time) mask and print its size and range."""
    mask = mask.detach().cpu().numpy()
    np.save(folder / (name + ".npy"), mask)
    print(f"  {name:34s} {str(mask.shape):11s} min {mask.min():10.4f}  max {mask.max():10.4f}")


def export_clip(path, setup, ap, detector, saliency, input_x_gradient, device):
    print(path)
    full_path = config.PROJECT_ROOT / path
    # 'LJ004-0082.wav_vocoded_0-1000.wav' -> 'LJ004-0082_0-1000', 'LJ041-0009.wav' -> 'LJ041-0009'
    folder_name = full_path.name.replace(".wav_vocoded_", "_").replace(".wav", "")
    folder = THIS_DIR / config.setup_key(setup) / folder_name
    folder.mkdir(parents=True, exist_ok=True)

    # The audio
    shutil.copyfile(full_path, folder / "audio_original.wav")
    waveform, _ = ap.load_audio(full_path)            # (80000,)
    np.save(folder / "audio_16k.npy", waveform.numpy())
    waveform = waveform.unsqueeze(0).to(device)       # (1, 80000)

    _, magnitude, phase = ap.compute_stft(waveform)   # (1, 513, 313)
    freq_bins = magnitude.shape[1]
    time_frames = magnitude.shape[2]

    with torch.no_grad():
        p_fake = torch.sigmoid(detector(magnitude, phase)).item()
    print(f"  detector p(fake) = {p_fake:.4f}")

    # Ground truth
    get_true_mask = GROUND_TRUTH[setup.localization]
    save_mask(folder, "mask_ground_truth", get_true_mask(path, (freq_bins, time_frames)))

    # Decoder
    if setup.localization == "time":
        old_mask = old_decoder_mask(waveform, setup.mask_ckpt, device)  # (1, 1, 512, 248)
        resized = F.interpolate(
            old_mask, size=(freq_bins, time_frames), mode="bilinear", align_corners=True
        )                                                               # (1, 1, 513, 313)
        save_mask(folder, "mask_decoder_512x248", old_mask[0, 0])
        save_mask(folder, "mask_decoder_resized_513x313", resized[0, 0])
        save_mask(folder, "mask_ground_truth_512x248", get_true_mask(path, (512, 248)))
    else:
        decoder = build_explainer(setup, "optimized", detector, device=device)
        save_mask(folder, "mask_decoder", decoder(magnitude, phase, [path])[0])

    # Captum
    save_mask(folder, "mask_saliency", saliency(magnitude, phase, [path])[0])
    save_mask(folder, "mask_input_x_gradient", input_x_gradient(magnitude, phase, [path])[0])


def export_setup(setup_name, paths, device):
    setup = config.get_setup(setup_name)
    ap = AudioProcessor(device=device)
    detector = build_detector(setup, ap, device=device)
    saliency = build_explainer(setup, "saliency", detector, device=device)
    input_x_gradient = build_explainer(setup, "input-x-gradient", detector, device=device)
    for path in paths:
        export_clip(path, setup, ap, detector, saliency, input_x_gradient, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="e.g. cuda:1 or cpu")
    args = parser.parse_args()
    device = config.get_device(args.device)

    export_setup("freq-swapped", FREQ_FILES, device)
    export_setup("time-swapped", TIME_FILES, device)


if __name__ == "__main__":
    main()
