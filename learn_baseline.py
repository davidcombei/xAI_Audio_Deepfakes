"""Learn a class-neutral noise fill for the masked-out bins (see masking.py).

The fill is fresh white noise, shaped per clip by the clip's own long-term spectrum times one
learned gain per frequency bin:

    B[f, t] = exp(log_gain[f]) * rms_t(|X[f, t]|) * N[f, t]

log_gain (513 values) is the only thing trained. Random soft masks replace parts of fakes and of
their bona fide clips with B, and the detector's per-frame logits are pushed to 0 where the clip
was replaced. The detector (wav2vec2 hidden layer 9 + logistic regression) stays frozen.

    python learn_baseline.py --setup freq-swapped-single --device cuda:4

Saves output/baseline/<setup>.pt and <setup>.txt (loss per step, fully masked logit per class
on held-out clips).
"""

from __future__ import annotations

import argparse
import itertools
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from audio import AudioProcessor
from data import FakeAndRealDataset, collate_fake_and_real, one_thread_per_worker, read_paths
from masking import keep_masked_part
from models import TorchLogReg
from train_mask import keep_only_needed_wav2vec2_layers

BASELINE_DIR = config.PROJECT_ROOT / "output" / "baseline"


def baseline_path(setup):
    return BASELINE_DIR / f"{setup.name}.pt"


def frame_logits(ap, logreg, waveforms):
    """(batch, 80000) -> (batch, frames). The detector's logit is the mean over the frames."""
    features = ap.extract_features(waveforms)        # (batch, frames, 1920)
    return logreg.linear(features).squeeze(-1)


class NoiseBaseline(nn.Module):
    def __init__(self, audio_processor, init_db=-15.0):
        super().__init__()
        self.ap = audio_processor
        self.log_gain = nn.Parameter(torch.full((config.N_FREQ_BINS,), init_db / 20 * math.log(10)))

    def sample(self, x):
        """A fresh fill for every clip: x (batch, 513, 313) complex -> same shape."""
        n_samples = config.SAMPLING_RATE * config.AUDIO_LENGTH
        noise, _, _ = self.ap.compute_stft(torch.randn(x.shape[0], n_samples, device=x.device))
        noise = noise / math.sqrt(config.WIN_LENGTH)
        spectrum = x.abs().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)   # (batch, 513, 1)
        return self.log_gain.exp()[None, :, None] * spectrum * noise


def load_baseline(setup, audio_processor, device):
    baseline = NoiseBaseline(audio_processor).to(device)
    baseline.load_state_dict(torch.load(baseline_path(setup), map_location=device))
    return baseline.eval().requires_grad_(False)


def random_masks(batch, n_freq, n_time, device, p_full=0.25):
    """Soft blob masks (batch, n_freq, n_time), 1 = keep. A share p_full is all zeros."""
    masks = []
    for _ in range(batch):
        if torch.rand(()) < p_full:
            masks.append(torch.zeros(n_freq, n_time, device=device))
            continue
        grid_freq = int(torch.randint(2, 9, ()))
        grid_time = int(torch.randint(3, 25, ()))
        blobs = torch.rand(1, 1, grid_freq, grid_time, device=device)
        blobs = F.interpolate(blobs, size=(n_freq, n_time), mode="bicubic", align_corners=False)[0, 0]
        kept_share = torch.rand((), device=device)
        threshold = torch.quantile(blobs.flatten(), 1.0 - kept_share)
        softness = 0.01 + 0.09 * torch.rand((), device=device)
        masks.append(torch.sigmoid((blobs - threshold) / softness))
    return torch.stack(masks)


def masked_share_per_frame(mask, n_frames):
    """(batch, n_freq, n_time) -> (batch, n_frames): share of each wav2vec2 frame that is replaced."""
    share = (1.0 - mask).mean(dim=1, keepdim=True)
    return F.adaptive_avg_pool1d(share, n_frames)[:, 0]


def neutrality_loss(logits, masked_share, power=2.0, w_pooled=1.0):
    """logits, masked_share: (batch, frames). Replaced frames should score 0, each one and pooled."""
    weight = masked_share.pow(power)
    per_frame = (weight * logits.pow(2)).sum() / weight.sum().clamp_min(1e-6)
    pooled = (weight * logits).sum(dim=1) / weight.sum(dim=1).clamp_min(1e-6)
    has_mask = (weight.sum(dim=1) > 1e-6).float()
    per_clip = (has_mask * pooled.pow(2)).sum() / has_mask.sum().clamp_min(1.0)
    return per_frame + w_pooled * per_clip


def envelope_regulariser(log_gain, w_level=1e-3, w_smooth=1e-2):
    level = log_gain.exp().pow(2).mean()
    smooth = (log_gain[1:] - log_gain[:-1]).pow(2).mean()
    return w_level * level + w_smooth * smooth


@torch.no_grad()
def fully_masked_logits(ap, logreg, baseline, loader, device, draws=4):
    """Detector logit of fill-only clips, per class: {"fake": (n,), "real": (n,)}."""
    logits = {"fake": [], "real": []}
    for fakes, reals, _ in loader:
        for label, waveforms in (("fake", fakes), ("real", reals)):
            x, _, _ = ap.compute_stft(waveforms.to(device))
            per_draw = []
            for _ in range(draws):
                filled = ap.compute_invert_stft(baseline.sample(x))
                per_draw.append(frame_logits(ap, logreg, filled).mean(dim=1))
            logits[label].append(torch.stack(per_draw).mean(dim=0).cpu())
    return {label: torch.cat(values) for label, values in logits.items()}


def train(setup, steps=2000, batch_size=8, lr=3e-2, device=None, workers=4, eval_clips=200, seed=0):
    setup = config.get_setup(setup) if isinstance(setup, str) else setup
    if setup.real_dir is None:
        raise ValueError(f"setup {setup.name!r} has no bona fide clips (real_dir is None)")
    device = config.get_device(device)
    ap = AudioProcessor(device=device)
    keep_only_needed_wav2vec2_layers(ap)
    logreg = TorchLogReg(setup.detector_ckpt).to(device)
    baseline = NoiseBaseline(ap).to(device)

    paths = read_paths(setup.filelist)
    random.Random(seed).shuffle(paths)
    eval_paths, train_paths = paths[:eval_clips], paths[eval_clips:]

    def make_loader(clip_paths, shuffle):
        dataset = FakeAndRealDataset(clip_paths, ap, root=config.PROJECT_ROOT, real_dir=setup.real_dir)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fake_and_real,
                          num_workers=workers, worker_init_fn=one_thread_per_worker)

    loader = make_loader(train_paths, shuffle=True)
    optimizer = torch.optim.Adam(baseline.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = baseline_path(setup).with_suffix(".txt")
    log = open(log_path, "w")

    def report(line):
        print(line)
        log.write(line + "\n")
        log.flush()

    report(f"{setup.name}: {len(train_paths)} training fakes + their bona fide clips, "
           f"{steps} steps x batch {2 * batch_size}")
    batches = itertools.chain.from_iterable(itertools.repeat(loader))
    for step in range(1, steps + 1):
        fakes, reals, _ = next(batches)
        waveforms = torch.cat([fakes, reals]).to(device)               # (2 * batch, 80000)
        with torch.no_grad():
            x, _, _ = ap.compute_stft(waveforms)                       # (2 * batch, 513, 313)
        mask = random_masks(x.shape[0], x.shape[1], x.shape[2], device)
        blended = keep_masked_part(mask, x, baseline.sample(x), "real")
        logits = frame_logits(ap, logreg, ap.compute_invert_stft(blended))
        neutrality = neutrality_loss(logits, masked_share_per_frame(mask, logits.shape[1]))
        loss = neutrality + envelope_regulariser(baseline.log_gain)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 50 == 0 or step == 1:
            report(f"step {step}: neutrality={neutrality.item():.4f}, total={loss.item():.4f}")

    baseline.eval().requires_grad_(False)
    torch.save(baseline.state_dict(), baseline_path(setup))
    report(f"saved {baseline_path(setup)}")

    logits = fully_masked_logits(ap, logreg, baseline, make_loader(eval_paths, shuffle=False), device)
    report(f"fully masked logit on {len(eval_paths)} held-out clips (0 = neutral):")
    for label, values in logits.items():
        report(f"  {label}: {values.mean().item():+.3f} +- {values.std().item():.3f}  "
               f"(p_fake = {torch.sigmoid(values).mean().item():.3f})")
    log.close()
    return baseline


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setup", choices=config.SETUP_NAMES, required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8, help="fakes per step; their bona fide clips double it")
    p.add_argument("--lr", type=float, default=3e-2)
    p.add_argument("--device", default=None, help="e.g. cuda:4")
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers that load the wavs")
    p.add_argument("--eval-clips", type=int, default=200, help="held-out fakes for the neutrality check")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.setup, steps=args.steps, batch_size=args.batch_size, lr=args.lr, device=args.device,
          workers=args.workers, eval_clips=args.eval_clips)
