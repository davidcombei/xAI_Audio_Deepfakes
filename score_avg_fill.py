"""What does a setup's detector say about the avg fill itself?

With avg fill, a masked-out bin gets the setup's average magnitude spectrogram (made by
compute_avg_spectrogram.py). If the mask removes everything, the detector sees only the fill.
This script scores that "everything masked" input, so the fill's own p(fake) is known:

  avg fill, own phase   average fake magnitude with each fake's own phase, for every fake in
                        setup.filelist. This is exactly what evaluate.py --fill avg feeds the
                        detector when the mask is all zeros (masking.avg_fill_spectrogram).
  avg fill, GL phase    average fake magnitude with one Griffin-Lim phase (one fixed clip)
  avgreal fill          average bona fide magnitude with Griffin-Lim phase, the fixed clip that
                        evaluate.py --fill avgreal uses (output/avg_spectrogram/<setup>_real_fill.npy)

For comparison it also prints the scores of the real fakes and bona fide clips, read from
compute_eer.py's csv (logs/evaluation_avgfill/<setup>__eer_scores.csv) when that file exists.

Every input goes through the same path as in evaluate.py: ISTFT -> wav2vec2 -> logistic regression.

    python score_avg_fill.py -s realistic --device cuda:2

Saves:
  output/avg_spectrogram/<setup>_fill_scores.csv   p(fake) of every "avg fill, own phase" clip
  output/avg_spectrogram/<setup>_gl.wav            the "avg fill, GL phase" clip, to listen to
"""

import csv
import math

import click
import numpy as np
import scipy.io.wavfile
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from audio import AudioProcessor
from data import AudioDataset, one_thread_per_worker, read_paths
from detector import build_detector
from masking import AVG_SPECTROGRAM_DIR, avg_fill_spectrogram, avg_real_fill_path, load_avg_magnitude


def griffin_lim(ap, magnitude, n_iter=500, momentum=0.99, seed=0):
    """Same as compute_avg_spectrogram.griffin_lim (copied: that file needs soundfile, which .venv lacks).

    (513, 313) magnitude -> (80000,) waveform, fixed seed, fast Griffin-Lim with the pipeline's own STFT.
    """
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


def score_spectrogram(detector, spectrogram):
    """p(fake) of complex spectrograms (batch, 513, 313) -> numpy array (batch,)."""
    with torch.no_grad():
        logits = detector(spectrogram.abs(), spectrogram.angle())
    return torch.sigmoid(logits).cpu().numpy()


def print_scores(name, scores):
    scores = np.asarray(scores)
    print(f"{name}:")
    print(f"  mean p(fake): {scores.mean():.4f}")
    print(f"  median p(fake): {np.median(scores):.4f}")
    print(f"  called fake (p > 0.5): {np.mean(scores > 0.5):.4f}")
    print(f"  n clips: {len(scores)}")


def read_eer_scores(setup):
    """p(fake) of the real fakes and bona fide clips from compute_eer.py, or None if not run yet."""
    path = config.PROJECT_ROOT / "logs" / "evaluation_avgfill" / f"{setup.name}__eer_scores.csv"
    if not path.exists():
        return None, None
    fakes = []
    reals = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["label"] == "1":
                fakes.append(float(row["p_fake"]))
            else:
                reals.append(float(row["p_fake"]))
    return fakes, reals


@click.command()
@click.option("-s", "--setup", "setup_name", type=click.Choice(config.SETUP_NAMES), required=True)
@click.option("-b", "--batch-size", default=8, show_default=True)
@click.option("--device", default=None, help="e.g. cuda:2")
@click.option("--workers", default=4, show_default=True, help="DataLoader workers that load the wavs")
@click.option("--max-clips", default=None, type=int, help="only the first N fakes (default: all)")
def main(setup_name, batch_size, device, workers, max_clips):
    setup = config.get_setup(setup_name)
    device = config.get_device(device)
    ap = AudioProcessor(device=device)
    detector = build_detector(setup, ap, device=device)
    avg_magnitude = load_avg_magnitude(setup, device)            # (513, 313)

    # 1. average fake magnitude + Griffin-Lim phase: one fixed clip
    gl_waveform = griffin_lim(ap, avg_magnitude)                 # (80000,)
    gl_spectrogram, _, _ = ap.compute_stft(gl_waveform.unsqueeze(0))   # (1, 513, 313)
    gl_score = score_spectrogram(detector, gl_spectrogram)
    scipy.io.wavfile.write(AVG_SPECTROGRAM_DIR / f"{setup.name}_gl.wav", config.SAMPLING_RATE,
                           gl_waveform.cpu().numpy().astype(np.float32))

    # 2. average bona fide magnitude + Griffin-Lim phase (the avgreal fill): one fixed clip
    avgreal_score = None
    if avg_real_fill_path(setup).exists():
        avgreal_spectrogram = torch.from_numpy(np.load(avg_real_fill_path(setup))).to(device)
        avgreal_score = score_spectrogram(detector, avgreal_spectrogram.unsqueeze(0))

    # 3. average fake magnitude + each fake's own phase, for every fake
    paths = read_paths(setup.filelist)
    if max_clips is not None:
        paths = paths[:max_clips]
    loader = DataLoader(
        AudioDataset(paths, ap, root=config.PROJECT_ROOT, device="cpu"),
        batch_size=batch_size, num_workers=workers, worker_init_fn=one_thread_per_worker,
    )
    own_phase_scores = []
    for waveforms, _ in tqdm(loader, desc=f"{setup.name} avg fill, own phase", ascii=True, mininterval=30):
        x_fake, _, _ = ap.compute_stft(waveforms.to(device))         # (batch, 513, 313) complex
        fill = avg_fill_spectrogram(avg_magnitude, x_fake)           # (batch, 513, 313) complex
        own_phase_scores.extend(score_spectrogram(detector, fill).tolist())

    out_csv = AVG_SPECTROGRAM_DIR / f"{setup.name}_fill_scores.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "p_fake_avg_fill_own_phase"])
        for path, score in zip(paths, own_phase_scores):
            writer.writerow([path, f"{score:.6f}"])

    print(f"\nsetup: {setup.name}")
    print(f"detector: {setup.detector_ckpt.name}\n")
    fakes, reals = read_eer_scores(setup)
    if fakes is not None:
        print_scores("real fakes", fakes)
        print_scores("bona fide clips", reals)
    print_scores("avg fill, own phase", own_phase_scores)
    print("avg fill, GL phase:")
    print(f"  p(fake): {gl_score[0]:.4f}")
    if avgreal_score is not None:
        print("avgreal fill:")
        print(f"  p(fake): {avgreal_score[0]:.4f}")
    print(f"\nsaved {out_csv}")


if __name__ == "__main__":
    main()
