"""Optimise one fixed fill waveform that every setup's detector treats as neutral (see masking.py).

The fill is a single 5 s waveform, the same for every clip and every setup. Its STFT replaces the
masked-out bins:

    blend = m * X + (1 - m) * X_fill

It is trained on fakes and bona fide clips of all three setups, each scored by its own setup's
detector (wav2vec2 hidden layer 9 + logistic regression, frozen), under random masks (blobs,
frequency bands, time segments, constant levels, everything removed). The detector's logit of a
blend should be what the kept part alone is worth:

    target = kept * z_base + kept_band * (z_clip - z_base)

    bona fide clip   z_base = z_clip = its own logit         -> kept * z_real
    swap fake        z_base = logit of its bona fide clip, kept_band = kept share of the swapped band
    realistic fake   z_base = 0, the whole clip is the "band" -> kept * z_fake

so the fill alone scores 0, a bona fide clip stays bona fide in proportion to what is kept, and a
swap fake only stops being fake when its swapped band is removed. Targets are set per wav2vec2
frame and for the clip.

    python learn_neutral_fill.py --name neutral_fill_all --device cuda:4
    python learn_neutral_fill.py --name neutral_fill_realistic --setups realistic --pairs-per-setup 12

--setups limits the clips and detectors to some setups (default: all three, one general fill).
A fill named neutral_fill_<setup> is what --fill optimized uses for that setup.

Fine-tuning on a U-Net's masks: once train_mask.py --fill optimized has produced a U-Net, its
masks no longer look like the random ones. --init starts from a saved fill and --mask-ckpt mixes
the frozen U-Net's masks (m and 1 - m, the two inputs of the LMAC loss) into the random masks:

    python learn_neutral_fill.py --name neutral_fill_freq-swapped_ft --setups freq-swapped \
        --init output/neutral_fill/neutral_fill_freq-swapped.wav \
        --mask-ckpt runs/freq-swapped_optimizedfill/<checkpoint>.pth --steps 1000 --lr 3e-3

--mask-ckpt takes several checkpoints (one is picked per step), so a fill can be tuned on the
masks of every U-Net the bootstrap produced (bootstrap_fill.sh).

Saves output/neutral_fill/<name>.wav (float32, 16 kHz), <name>.npy (complex64 STFT, 513 x 313)
and <name>.txt (training log and held-out results, also for the starting fill).
"""

from __future__ import annotations

import argparse
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

import config
from audio import AudioProcessor
from compute_avg_spectrogram import griffin_lim
from data import read_paths, real_path, utterance_id
from masking import AVG_SPECTROGRAM_DIR, NEUTRAL_FILL_DIR
from metrics import BOUNDARY_HZ, LocalizationEvaluator, ground_truth_freq
from models import TorchLogReg, load_mask_predictor
from train_mask import keep_only_needed_wav2vec2_layers

SETUPS = ("realistic", "freq-swapped", "freq-swapped-single")
FILL_DIR = NEUTRAL_FILL_DIR
EVAL_UTTERANCES = "LJ050"
LOGIT_SCALE = 100.0
SHAPE = (config.N_FREQ_BINS, config.N_TIME_FRAMES)


def split_paths(setup, n_train, n_eval, seed):
    """Fakes for training and for evaluation. Swap setups: one fake per LJSpeech utterance, and the
    LJ050 utterances are only ever used for evaluation."""
    rng = random.Random(seed)
    paths = read_paths(setup.filelist)
    if setup.localization is None:
        rng.shuffle(paths)
        return paths[n_eval:n_eval + n_train], paths[:n_eval]
    by_utterance = {}
    for path in paths:
        by_utterance.setdefault(utterance_id(path), []).append(path)
    train, held_out = [], []
    for utterance in sorted(by_utterance):
        fake = rng.choice(by_utterance[utterance])
        (held_out if utterance.startswith(EVAL_UTTERANCES) else train).append(fake)
    rng.shuffle(train)
    rng.shuffle(held_out)
    return train[:n_train], held_out[:n_eval]


def frame_logits(ap, logreg, waveforms):
    """(batch, 80000) -> (batch, frames). The detector's logit is the mean over the frames."""
    return logreg.linear(ap.extract_features(waveforms)).squeeze(-1)


@torch.no_grad()
def load_pool(setup, paths, ap, logreg):
    def load(path):
        fake, _ = ap.load_audio(config.PROJECT_ROOT / path)
        real, _ = ap.load_audio(real_path(path, setup.real_dir))
        return fake, real

    with ThreadPoolExecutor(16) as pool:
        clips = list(pool.map(load, paths))
    fakes = torch.stack([fake for fake, _ in clips])                  # (n, 80000)
    reals = torch.stack([real for _, real in clips])
    if setup.localization is None:
        band = torch.ones(len(paths), SHAPE[0])
    else:
        band = torch.stack([ground_truth_freq(path, SHAPE, BOUNDARY_HZ)[:, 0] for path in paths])

    def logits(waveforms):
        chunks = [frame_logits(ap, logreg, waveforms[i:i + 16].to(ap.device)).cpu()
                  for i in range(0, len(waveforms), 16)]
        return torch.cat(chunks)                                      # (n, frames)

    z_fake, z_real = logits(fakes), logits(reals)
    z_base = torch.zeros_like(z_fake) if setup.localization is None else z_real
    return {"paths": paths, "fake": fakes, "real": reals, "band": band,
            "z_fake": z_fake, "z_real": z_real, "z_base": z_base, "logreg": logreg}


def rows(pool, index, device):
    """Fakes then their bona fide clips: waveforms, own logits, base logits, band (see module docstring)."""
    waveforms = torch.cat([pool["fake"][index], pool["real"][index]]).to(device)
    z_clip = torch.cat([pool["z_fake"][index], pool["z_real"][index]]).to(device)
    z_base = torch.cat([pool["z_base"][index], pool["z_real"][index]]).to(device)
    band = torch.cat([pool["band"][index], pool["band"][index]]).to(device)
    return waveforms, z_clip, z_base, band


def soft_region(n, softness, device):
    width = random.uniform(0.03, 0.6) * n
    low = random.uniform(0, n - width)
    position = torch.arange(n, device=device)
    return torch.sigmoid((position - low) / softness) * torch.sigmoid((low + width - position) / softness)


def random_mask(device):
    """One soft mask (513, 313), 1 = keep."""
    kind = random.choices(["blobs", "freq", "time", "level", "full"], weights=[35, 30, 15, 10, 10])[0]
    if kind == "full":
        return torch.zeros(SHAPE, device=device)
    if kind == "level":
        return torch.full(SHAPE, random.random(), device=device)
    if kind == "blobs":
        blobs = torch.rand(1, 1, random.randint(2, 8), random.randint(3, 24), device=device)
        blobs = F.interpolate(blobs, size=SHAPE, mode="bicubic", align_corners=False)[0, 0]
        threshold = torch.quantile(blobs.flatten(), 1.0 - random.random())
        return torch.sigmoid((blobs - threshold) / random.uniform(0.01, 0.1))
    axis = 0 if kind == "freq" else 1
    region = soft_region(SHAPE[axis], random.uniform(0.3, 4.0), device)
    if random.random() < 0.5:
        region = 1 - region
    return (region[:, None] if axis == 0 else region[None, :]).expand(SHAPE)


@torch.no_grad()
def unet_masks(ap, unet, waveforms):
    """The frozen U-Net's masks for these clips: (batch, 80000) -> (batch, 513, 313), 1 = keep."""
    masks = []
    for i in range(0, len(waveforms), 16):
        _, magnitude, _ = ap.compute_stft(waveforms[i:i + 16])
        masks.append(unet(magnitude.unsqueeze(1)).squeeze(1))
    return torch.cat(masks)


def training_masks(ap, waveforms, unets, unet_share):
    """Random masks. With U-Nets, a share of them is one U-Net's mask for the clip, or 1 - its mask."""
    n = len(waveforms)
    masks = torch.stack([random_mask(ap.device) for _ in range(n)])
    if not unets:
        return masks
    learned = unet_masks(ap, random.choice(unets), waveforms)
    inverted = torch.rand(n, device=ap.device) < 0.5
    learned = torch.where(inverted[:, None, None], 1 - learned, learned)
    from_unet = torch.rand(n, device=ap.device) < unet_share
    return torch.where(from_unet[:, None, None], learned, masks)


def pool_frames(per_stft_frame, n_frames):
    return F.adaptive_avg_pool1d(per_stft_frame[:, None, :], n_frames)[:, 0]


def frame_targets(mask, band, z_clip, z_base):
    """mask (batch, 513, 313), band (batch, 513), logits (batch, frames) -> targets (batch, frames)."""
    n_frames = z_clip.shape[1]
    kept = pool_frames(mask.mean(dim=1), n_frames)
    kept_band = pool_frames((mask * band[:, :, None]).sum(dim=1) / band.sum(dim=1, keepdim=True), n_frames)
    target = kept * z_base + kept_band * (z_clip - z_base)
    zero = torch.zeros_like(target)
    low = torch.minimum(torch.minimum(z_clip, z_base), zero)
    high = torch.maximum(torch.maximum(z_clip, z_base), zero)
    return torch.clamp(target, low, high)


def rms(waveforms):
    return waveforms.pow(2).mean(dim=-1).sqrt()


def blend(ap, waveforms, mask, fill_waveform, match_level):
    """Waveforms (batch, 80000) whose masked-out bins are replaced by the fill."""
    x, _, _ = ap.compute_stft(waveforms)
    fill, _, _ = ap.compute_stft(fill_waveform)
    fill = fill.unsqueeze(0)
    if match_level:
        fill = fill * (rms(waveforms) / rms(fill_waveform))[:, None, None]
    return ap.compute_invert_stft(mask * x + (1 - mask) * fill)


def starting_fill(ap):
    """Griffin-Lim waveform of the mean of the LJSpeech and M-AILABS average bona fide spectrograms."""
    magnitudes = [np.load(AVG_SPECTROGRAM_DIR / f"{name}_real.npy") for name in ("freq-swapped", "realistic")]
    magnitude = torch.tensor(np.mean(magnitudes, axis=0), device=ap.device)
    return griffin_lim(ap, magnitude, n_iter=200)


@torch.no_grad()
def evaluate(pools, ap, fill_waveform, match_level, report, unets=(), masks_per_clip=4, seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    device = ap.device
    alone = ap.extract_features(fill_waveform.unsqueeze(0))
    for name, pool in pools.items():
        logreg = pool["logreg"]
        n = len(pool["paths"])
        report(f"  {name} ({n} held-out fakes + their bona fide clips)")
        report(f"    fill alone: logit {logreg.linear(alone).mean().item():+.1f}")

        def clip_logits(waveforms, mask):
            out = []
            for i in range(0, len(waveforms), 16):
                blended = blend(ap, waveforms[i:i + 16], mask[i:i + 16], fill_waveform, match_level)
                out.append(frame_logits(ap, logreg, blended).mean(dim=1))
            return torch.cat(out)

        waveforms, z_clip, z_base, band = rows(pool, torch.arange(n), device)
        errors = []
        for _ in range(masks_per_clip):
            mask = torch.stack([random_mask(device) for _ in range(2 * n)])
            target = frame_targets(mask, band, z_clip, z_base).mean(dim=1)
            errors.append(clip_logits(waveforms, mask) - target)
        errors = torch.stack(errors)                                  # (masks, 2n)
        for label, part in (("fakes", errors[:, :n]), ("bona fide", errors[:, n:])):
            report(f"    random masks, {label}: logit - target = {part.mean().item():+.1f} "
                   f"(rmse {part.pow(2).mean().sqrt().item():.1f})")

        for number, unet in enumerate(unets, start=1):
            learned = unet_masks(ap, unet, waveforms)
            report(f"    U-Net {number} masks: mean value {learned[:n].mean().item():.2f} on fakes, "
                   f"{learned[n:].mean().item():.2f} on bona fide")
            if "freq" in name:
                localization = LocalizationEvaluator("freq")
                localization.evaluate_batch(learned[:n], audio_path=pool["paths"])
                found = localization.results()
                report(f"    U-Net {number} masks on fakes: AUC {found['Localization']:.3f}, share of the mask inside the "
                       f"swapped band {found['Localization RMA']:.3f} (chance {found['Localization RMA (chance)']:.3f})")
            for label, mask in (("kept", learned), ("removed", 1 - learned)):
                z = clip_logits(waveforms, mask)
                target = frame_targets(mask, band, z_clip, z_base).mean(dim=1)
                for clips, part in (("fakes", slice(0, n)), ("bona fide", slice(n, 2 * n))):
                    report(f"    U-Net {number} mask {label}, {clips}: logit {z[part].mean().item():+.1f}, "
                           f"target {target[part].mean().item():+.1f}, "
                           f"rmse {(z[part] - target[part]).pow(2).mean().sqrt().item():.1f}, "
                           f"called fake {(z[part] > 0).float().mean().item():.2f}")

        if "freq" not in name:
            continue
        fakes, reals = waveforms[:n], waveforms[n:]
        true_band = (band[:n] > 0.5).float()[:, :, None].expand(n, *SHAPE)
        wrong_band = true_band.roll(128, dims=1)
        cases = (
            ("fake untouched", fakes, torch.ones_like(true_band)),
            ("bona fide untouched", reals, torch.ones_like(true_band)),
            ("fake, swapped band -> fill", fakes, 1 - true_band),
            ("fake, another band -> fill", fakes, 1 - wrong_band),
            ("bona fide, same band -> fill", reals, 1 - true_band),
            ("fake, only swapped band kept", fakes, true_band),
            ("bona fide, only same band kept", reals, true_band),
        )
        for label, clips, mask in cases:
            z = clip_logits(clips, mask)
            report(f"    {label:31s} logit {z.mean().item():+7.1f} +- {z.std().item():5.1f}, "
                   f"called fake {(z > 0).float().mean().item():.2f}")


def train(name="neutral_fill", setups=SETUPS, steps=3000, pairs_per_setup=4, lr=1e-2, device=None,
          n_train=384, n_eval=64, match_level=False, init=None, mask_ckpt=None, unet_share=0.5,
          w_frame=0.1, seed=0):
    device = config.get_device(device)
    random.seed(seed)
    torch.manual_seed(seed)
    ap = AudioProcessor(device=device)
    keep_only_needed_wav2vec2_layers(ap)

    FILL_DIR.mkdir(parents=True, exist_ok=True)
    log = open(FILL_DIR / f"{name}.txt", "w")

    def report(line):
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    train_pools, eval_pools = {}, {}
    for setup_name in setups:
        setup = config.get_setup(setup_name)
        logreg = TorchLogReg(setup.detector_ckpt).to(device)
        train_paths, eval_paths = split_paths(setup, n_train, n_eval, seed)
        train_pools[setup_name] = load_pool(setup, train_paths, ap, logreg)
        eval_pools[setup_name] = load_pool(setup, eval_paths, ap, logreg)
        report(f"{setup_name}: {len(train_paths)} training and {len(eval_paths)} held-out fakes, "
               f"each with its bona fide clip")

    unets = []
    if mask_ckpt:
        if len(setups) != 1:
            raise ValueError("--mask-ckpt needs exactly one setup: a U-Net belongs to one setup")
        for number, ckpt in enumerate(mask_ckpt, start=1):
            unets.append(load_mask_predictor(ckpt, device).requires_grad_(False))
            report(f"U-Net {number}: {ckpt}")
        report(f"masks: {unet_share:.0%} from the frozen U-Nets (one picked per step), the rest random")

    if init is None:
        start = starting_fill(ap)
        report("starting fill (average bona fide spectrum, Griffin-Lim phase):")
    else:
        start = ap.load_audio(init)[0].to(device)
        report(f"starting fill ({init}):")
    start_rms = rms(start)
    unit_fill = (start / start_rms).clone().requires_grad_(True)
    evaluate(eval_pools, ap, start, match_level, report, unets)

    optimizer = torch.optim.Adam([unit_fill], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    def save(fill_waveform):
        spectrogram, _, _ = ap.compute_stft(fill_waveform)
        np.save(FILL_DIR / f"{name}.npy", spectrogram.to(torch.complex64).cpu().numpy())
        sf.write(FILL_DIR / f"{name}.wav", fill_waveform.cpu().numpy(), config.SAMPLING_RATE, subtype="FLOAT")

    for step in range(1, steps + 1):
        fill_waveform = unit_fill * start_rms
        blended, targets, sizes = [], [], []
        for pool in train_pools.values():
            index = torch.randint(len(pool["paths"]), (pairs_per_setup,))
            waveforms, z_clip, z_base, band = rows(pool, index, device)
            mask = training_masks(ap, waveforms, unets, unet_share)
            blended.append(blend(ap, waveforms, mask, fill_waveform, match_level))
            targets.append(frame_targets(mask, band, z_clip, z_base))
            sizes.append(len(waveforms))
        features = ap.extract_features(torch.cat(blended + [fill_waveform.unsqueeze(0)]))

        logits, alone = [], []
        for pool, part in zip(train_pools.values(), features[:-1].split(sizes)):
            logits.append(pool["logreg"].linear(part).squeeze(-1))
            alone.append(pool["logreg"].linear(features[-1]).mean())
        error = (torch.cat(logits) - torch.cat(targets)) / LOGIT_SCALE         # (clips, frames)
        clip_loss = error.mean(dim=1).pow(2).mean()
        frame_loss = error.pow(2).mean()
        alone_loss = (torch.stack(alone) / LOGIT_SCALE).pow(2).mean()
        loss = clip_loss + w_frame * frame_loss + alone_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 50 == 0 or step == 1:
            report(f"step {step}: clip={clip_loss.item():.4f}, frame={frame_loss.item():.4f}, "
                   f"alone={alone_loss.item():.4f}")
        if step % 500 == 0:
            save((unit_fill * start_rms).detach())

    fill_waveform = (unit_fill * start_rms).detach()
    save(fill_waveform)
    report(f"saved {FILL_DIR / name}.wav and .npy  rms {rms(fill_waveform).item():.4f}, "
           f"peak {fill_waveform.abs().max().item():.4f}")
    report("optimised fill:")
    evaluate(eval_pools, ap, fill_waveform, match_level, report, unets)
    log.close()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="neutral_fill", help="output/neutral_fill/<name>.{wav,npy,txt}")
    p.add_argument("--setups", nargs="+", choices=SETUPS, default=list(SETUPS))
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--pairs-per-setup", type=int, default=4, help="fake + bona fide pairs per setup and step")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--device", default=None, help="e.g. cuda:4")
    p.add_argument("--n-train", type=int, default=384, help="training pairs per setup")
    p.add_argument("--n-eval", type=int, default=64, help="held-out pairs per setup")
    p.add_argument("--match-level", action="store_true", help="scale the fill to each clip's RMS")
    p.add_argument("--init", default=None, help="saved fill .wav to start from (default: average bona fide spectrum)")
    p.add_argument("--mask-ckpt", nargs="+", default=None,
                   help="U-Net checkpoint(s) whose masks are mixed into the random ones")
    p.add_argument("--unet-share", type=float, default=0.5, help="share of the masks taken from the U-Net")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.name, setups=args.setups, steps=args.steps, pairs_per_setup=args.pairs_per_setup, lr=args.lr,
          device=args.device, n_train=args.n_train, n_eval=args.n_eval, match_level=args.match_level,
          init=args.init, mask_ckpt=args.mask_ckpt, unet_share=args.unet_share)
