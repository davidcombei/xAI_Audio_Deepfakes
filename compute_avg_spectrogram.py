"""Average magnitude spectrogram of one setup's training data, for avg fill (see masking.py).

For every clip in the setup's own training metadata (Setup.filelist):
    1. load the waveform exactly as training/evaluation do (16 kHz, cut or zero-padded to 5 s)
    2. compute its STFT -> magnitude spectrogram |X|, (513, 313)
Then the average over all clips, bin by bin:
    avg[f, t] = (|X_1[f, t]| + |X_2[f, t]| + ... + |X_N[f, t]|) / N

The waveforms are never averaged, only the spectrograms. The spectrograms are added to a running
sum instead of being kept in memory (40000 x 513 x 313 floats would be ~26 GB); the result is the
same number as stacking them all and taking the mean.

Every setup gets its own average, from its own metadata only:
    realistic            metadata/realistic-all.txt          (27171 clips)
    freq-swapped         metadata/freq-swap-all.txt          (40000 clips)
    freq-swapped-single  metadata/freq-swap-single-all.txt   ( 5000 clips)

    python compute_avg_spectrogram.py -s freq-swapped --device cuda:2

Saves output/avg_spectrogram/<setup>.npy, (513, 313) float32, and <setup>.txt (metadata file, clip count).

--source real averages the setup's unique bona fide clips instead (the real_dir counterparts of
the fakes) and turns the average into one fixed complex spectrogram with Griffin-Lim, for avgreal
fill. Saves <setup>_real.npy (average magnitude), <setup>_real_fill.npy (complex64, the fill),
<setup>_real_fill.wav and <setup>_real.txt.

    python compute_avg_spectrogram.py -s freq-swapped --source real --device cuda:2
"""

from __future__ import annotations

import math

import click
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from audio import AudioProcessor
from data import FakeAndRealDataset, collate_fake_and_real, one_thread_per_worker, read_paths, real_path
from masking import AVG_SPECTROGRAM_DIR, avg_real_fill_path, avg_spectrogram_path


def clip_paths(setup, source):
    paths = read_paths(setup.filelist)
    if source == "fake":
        return paths, config.PROJECT_ROOT
    if setup.real_dir is None:
        raise ValueError(f"setup {setup.name!r} has no bona fide clips (real_dir is None)")
    reals = sorted({str(real_path(p, setup.real_dir)) for p in paths})
    return reals, None


def compute_avg_spectrogram(setup, batch_size=32, device=None, workers=8, source="fake"):
    """(513, 313) average magnitude spectrogram over the setup's fakes or its unique bona fide clips."""
    device = config.get_device(device)
    ap = AudioProcessor(device=device)
    paths, root = clip_paths(setup, source)

    dataset = FakeAndRealDataset(paths, ap, root=root, real_dir=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fake_and_real,
                        num_workers=workers, worker_init_fn=one_thread_per_worker)

    total = None      # (513, 313) running sum of the magnitude spectrograms
    n_clips = 0
    for clips, _, _ in tqdm(loader, desc=f"avg {source} spectrogram {setup.name}", ascii=True):
        _, magnitude, _ = ap.compute_stft(clips.to(device))      # (batch, 513, 313)
        batch_sum = magnitude.double().sum(dim=0)                # (513, 313), float64 so the sum stays exact
        if total is None:
            total = batch_sum
        else:
            total = total + batch_sum
        n_clips = n_clips + magnitude.shape[0]

    if n_clips != len(paths):
        raise RuntimeError(f"read {n_clips} clips but expected {len(paths)}")
    avg = total / n_clips                                        # (513, 313)
    return avg.float().cpu().numpy(), n_clips


def griffin_lim(ap, magnitude, n_iter=500, momentum=0.99, seed=0):
    """Fixed-seed fast Griffin-Lim with the pipeline's own STFT: (513, 313) magnitude -> (80000,) waveform."""
    generator = torch.Generator().manual_seed(seed)
    angles = torch.exp(2j * math.pi * torch.rand(magnitude.shape, generator=generator, dtype=torch.float64))
    angles = angles.to(torch.complex64).to(magnitude.device)
    previous = torch.zeros_like(angles)
    for _ in range(n_iter):
        waveform = ap.compute_invert_stft((magnitude * angles).unsqueeze(0))
        rebuilt, _, _ = ap.compute_stft(waveform)
        rebuilt = rebuilt.squeeze(0)
        angles = rebuilt - previous * (momentum / (1 + momentum))
        angles = angles / (angles.abs() + 1e-16)
        previous = rebuilt
    return ap.compute_invert_stft((magnitude * angles).unsqueeze(0)).squeeze(0)


def save_real_fill(setup, avg, n_clips, device):
    ap = AudioProcessor(device=config.get_device(device))
    magnitude = torch.tensor(avg, device=ap.device)
    waveform = griffin_lim(ap, magnitude)
    fill, _, _ = ap.compute_stft(waveform)                       # (513, 313) complex
    error = (fill.abs() - magnitude).norm() / magnitude.norm()

    np.save(AVG_SPECTROGRAM_DIR / f"{setup.name}_real.npy", avg)
    out = avg_real_fill_path(setup)
    np.save(out, fill.to(torch.complex64).cpu().numpy())
    sf.write(out.with_suffix(".wav"), waveform.cpu().numpy(), config.SAMPLING_RATE)
    with open(AVG_SPECTROGRAM_DIR / f"{setup.name}_real.txt", "w") as f:
        f.write(f"setup: {setup.name}\n")
        f.write(f"bona fide clips: {n_clips} unique {setup.real_dir} counterparts of {setup.filelist}\n")
        f.write(f"shape: {avg.shape}\n")
        f.write("average of the magnitude spectrograms |STFT| (n_fft 1024, hop 256), 16 kHz, 5 s\n")
        f.write(f"fill: Griffin-Lim phase (500 iterations, seed 0), spectral convergence {error.item():.4f}\n")
    print(f"saved {out}  Griffin-Lim spectral convergence {error.item():.4f}")


@click.command()
@click.option("-s", "--setup", "setup_name", type=click.Choice(config.SETUP_NAMES), required=True)
@click.option("-b", "--batch-size", default=32, show_default=True)
@click.option("--device", default=None, help="e.g. cuda:2")
@click.option("--workers", default=8, show_default=True, help="DataLoader workers that load the wavs")
@click.option("--source", type=click.Choice(["fake", "real"]), default="fake", show_default=True,
              help="fake = average of the setup's fakes (avg fill); real = of its bona fide clips (avgreal fill)")
def main(setup_name, batch_size, device, workers, source):
    setup = config.get_setup(setup_name)
    avg, n_clips = compute_avg_spectrogram(setup, batch_size, device, workers, source)

    AVG_SPECTROGRAM_DIR.mkdir(parents=True, exist_ok=True)
    if source == "real":
        print(f"{setup.name}: average of {n_clips} bona fide magnitude spectrograms")
        save_real_fill(setup, avg, n_clips, device)
        return
    out = avg_spectrogram_path(setup)
    np.save(out, avg)
    with open(out.with_suffix(".txt"), "w") as f:
        f.write(f"setup: {setup.name}\n")
        f.write(f"metadata: {setup.filelist}\n")
        f.write(f"clips: {n_clips}\n")
        f.write(f"shape: {avg.shape}\n")
        f.write("average of the magnitude spectrograms |STFT| (n_fft 1024, hop 256), 16 kHz, 5 s\n")
    print(f"{setup.name}: average of {n_clips} magnitude spectrograms from {setup.filelist}")
    print(f"saved {out}  shape {avg.shape}  min {avg.min():.4f}  mean {avg.mean():.4f}  max {avg.max():.4f}")


if __name__ == "__main__":
    main()
