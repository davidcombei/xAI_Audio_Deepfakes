"""Make 2-3 kHz band swaps whose band follows the detector's mel filters.

The existing freq swaps (LJSpeech_freq_swap/*_vocoded_2000-3000.wav) copy
every STFT bin between 2000 and 3000 Hz from the vocoded clip into the real
clip. That band has hard edges that do not line up with the 80 mel filters
the detector uses: the filters overlap (triangles), so 2000 Hz and 3000 Hz
fall in the middle of some of them.

Here the vocoded part is weighted by the mel filters themselves:

    W(f)     = sum of the mel filters 43..52   (the filters centred between 2000 and 3000 Hz)
    combined = (1 - W(f)) * STFT_real + W(f) * STFT_vocoded

Neighbouring HTK mel triangles add up to 1, so W(f) is:

    0              below 1967 Hz
    rises to 1     1967 -> 2052 Hz   (the left half of filter 43)
    1              2052 -> 2941 Hz   (the centres of filters 43 and 52)
    falls to 0     2941 -> 3056 Hz   (the right half of filter 52)
    0              above 3056 Hz

So the swapped region is exactly "mel filters 43..52" and the edges are
smooth ramps instead of hard cuts.

Everything else is the same as the original swaps (checked: re-running the
original recipe on the files below reproduces LJSpeech_freq_swap/ up to
16-bit rounding, max error 2.5e-5):
  * real clip:     LJSpeech_vocoded22K/<id>.wav            (22050 Hz)
  * vocoded clip:  LJSpeech_hifigan/<id>.wav_vocoded.wav    with the first 1330 samples
                   (HiFi-GAN latency) dropped, so it lines up with the real clip
  * STFT:          n_fft 1024, hop 256, rectangular window (torch default)

--window hann uses a Hann window in the STFT and the ISTFT instead. The
rectangular window leaks the swap into every frequency: outside 1.9-3.15 kHz
the fake differs from the real clip by 1e-4 to 0.3 of the real energy (0.33
below 100 Hz, where the real clip has little energy). With the Hann window
that difference is below 1e-6 everywhere outside the band (checked on
LJ001-0001), so the only change is the vocoded mel band itself.

Output: logReg_coeff_logMel/LJSpeech_freq_swap_mel/<id>.wav_vocoded_mel43-52.wav       (--window rect)
        logReg_coeff_logMel/LJSpeech_freq_swap_mel_hann/<id>.wav_vocoded_mel43-52.wav  (--window hann)

Run from the repo root:

    python3 logReg_coeff_logMel/make_mel_swaps.py --window rect
    python3 logReg_coeff_logMel/make_mel_swaps.py --window hann
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: config.py

import argparse
import math
import warnings

import torch
import torchaudio
from tqdm import tqdm

import config

warnings.filterwarnings("ignore", message="A window was not provided")  # rectangular window on purpose

EXPERIMENT_DIR = Path(__file__).resolve().parent
OUT_DIRS = {
    "rect": EXPERIMENT_DIR / "LJSpeech_freq_swap_mel",
    "hann": EXPERIMENT_DIR / "LJSpeech_freq_swap_mel_hann",
}

REAL_DIR = config.TRAIN_AUDIO_DIR       # LJSpeech_vocoded22K
VOCODED_DIR = config.VOCODED_DIR        # LJSpeech_hifigan
HIFIGAN_LATENCY = 1330                  # samples, same as vocoder.py

SWAP_SR = 22050                         # sample rate of the wav files
SWAP_N_FFT = 1024
SWAP_HOP = 256

# The detector's mel filter bank (simple_classifier/mel_detector.py): torchaudio
# MelSpectrogram defaults = HTK mel scale, 0..8000 Hz, no normalisation.
N_MELS = config.N_MELS                  # 80
MEL_F_MIN = 0.0
MEL_F_MAX = config.SAMPLING_RATE / 2    # 8000 Hz
BAND_LOW_HZ = 2000
BAND_HIGH_HZ = 3000


def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_points():
    """The 82 corner frequencies (Hz) of the 80 triangular mel filters.

    Filter k starts at point k, peaks at point k + 1 and ends at point k + 2.
    """
    low = hz_to_mel(MEL_F_MIN)
    high = hz_to_mel(MEL_F_MAX)
    points = []
    for i in range(N_MELS + 2):
        points.append(mel_to_hz(low + i * (high - low) / (N_MELS + 1)))
    return points


def swapped_mel_bands():
    """Indices of the mel filters whose centre is between 2000 and 3000 Hz (43..52)."""
    points = mel_points()
    bands = []
    for k in range(N_MELS):
        centre = points[k + 1]
        if BAND_LOW_HZ <= centre < BAND_HIGH_HZ:
            bands.append(k)
    return bands


def triangle(f, left, centre, right):
    """Value of one mel filter (peak 1) at frequency f."""
    if f <= left or f >= right:
        return 0.0
    if f <= centre:
        return (f - left) / (centre - left)
    return (right - f) / (right - centre)


def swap_weights():
    """W(f) for every STFT bin of the 22050 Hz swap STFT: (513,), values in [0, 1]."""
    points = mel_points()
    bands = swapped_mel_bands()
    n_bins = SWAP_N_FFT // 2 + 1
    weights = torch.zeros(n_bins)
    for b in range(n_bins):
        f = b * SWAP_SR / SWAP_N_FFT
        total = 0.0
        for k in bands:
            total += triangle(f, points[k], points[k + 1], points[k + 2])
        weights[b] = min(total, 1.0)
    return weights


def load_mono(path):
    audio, sr = torchaudio.load(str(path))
    if sr != SWAP_SR:
        raise ValueError(f"{path}: expected {SWAP_SR} Hz, got {sr}")
    return audio[0]


def make_one(real, vocoded, weights, window_name="rect"):
    """real, vocoded: waveforms at 22050 Hz -> mel-band-swapped waveform."""
    if window_name == "hann":
        window = torch.hann_window(SWAP_N_FFT)
    else:
        window = None  # torch default: rectangular, like the original swaps
    stft_real = torch.stft(real, n_fft=SWAP_N_FFT, hop_length=SWAP_HOP, window=window, return_complex=True)
    stft_voc = torch.stft(vocoded, n_fft=SWAP_N_FFT, hop_length=SWAP_HOP, window=window, return_complex=True)
    n_frames = min(stft_real.shape[1], stft_voc.shape[1])
    stft_real = stft_real[:, :n_frames]                             # (513, frames)
    stft_voc = stft_voc[:, :n_frames]                               # (513, frames)

    w = weights.view(-1, 1)                                         # (513, 1)
    combined = (1 - w) * stft_real + w * stft_voc
    return torch.istft(combined, n_fft=SWAP_N_FFT, hop_length=SWAP_HOP, window=window)


def utterance_ids():
    """The utterances that have a 2-3 kHz swap in LJSpeech_freq_swap/ (one per utterance)."""
    ids = []
    folder = config.SETUPS["freq-swapped"].data_dir
    for wav in sorted(folder.glob("*.wav_vocoded_2000-3000.wav")):
        ids.append(wav.name[:10])  # 'LJ001-0001.wav_vocoded_2000-3000.wav' -> 'LJ001-0001'
    return ids


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--window", choices=["rect", "hann"], default="rect", help="STFT window of the swap")
    args = p.parse_args()
    out_dir = OUT_DIRS[args.window]
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = swap_weights()
    bands = swapped_mel_bands()
    points = mel_points()
    print(f"mel bands {bands[0]}..{bands[-1]} ({len(bands)} bands), "
          f"W = 1 from {points[bands[0] + 1]:.0f} to {points[bands[-1] + 1]:.0f} Hz, "
          f"W > 0 from {points[bands[0]]:.0f} to {points[bands[-1] + 2]:.0f} Hz")

    ids = utterance_ids()
    for uid in tqdm(ids, ascii=True, desc="mel swaps", mininterval=30):
        out_path = out_dir / f"{uid}.wav_vocoded_mel{bands[0]}-{bands[-1]}.wav"
        if out_path.exists():
            continue
        real = load_mono(REAL_DIR / f"{uid}.wav")
        vocoded = load_mono(VOCODED_DIR / f"{uid}.wav_vocoded.wav")[HIFIGAN_LATENCY:]
        wav = make_one(real, vocoded, weights, args.window)
        torchaudio.save(str(out_path), wav.unsqueeze(0), SWAP_SR)
    print(f"{len(ids)} clips -> {out_dir}")


if __name__ == "__main__":
    main()
