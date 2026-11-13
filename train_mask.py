"""Train the U-Net mask predictor with the LMAC loss.

By default the masked-out bins are filled with the bona fide clip the fake was
made from ("real fill", see masking.py); --fill avg fills them with the setup's
average spectrogram (run compute_avg_spectrogram.py first); --fill avgreal with the
average bona fide spectrogram (compute_avg_spectrogram.py --source real); --fill optimized with
the fill optimised to be neutral for the setup's detector (learn_neutral_fill.py); --fill zero gives the
original zero-masking loss. The U-Net only sees the fake clip's 513 x 313 STFT magnitude;
the bona fide clip is only used inside the loss. Train on all files of the setup.

Run from the repo root, e.g. on GPUs 0-4:

    accelerate launch --multi_gpu --gpu_ids 0,1,2,3,4 --num_processes 5 \\
        train_mask.py --setup freq-swapped --epochs 40 --batch-size 15 --resume
"""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from audio import AudioProcessor
from data import FakeAndRealDataset, collate_fake_and_real, one_thread_per_worker, read_paths
from losses import LMACLoss
from masking import FILLS, check_fill, fill_spectrogram, load_fill_reference
from models import UNet, TorchLogReg

# The loss backprops through the frozen wav2vec2 via ISTFT features; the math
# SDP kernel keeps that attention backward numerically stable.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def keep_only_needed_wav2vec2_layers(ap):
    """Drop the wav2vec2 layers above the one the detector reads.

    The checkpoint only has trained weights for the first 9 layers but its
    config declares 48, so the extra layers are random and never used. The
    detector reads hidden_states[9] (the input of layer 9), so keeping layers
    0..9 gives exactly the same features and makes training about 4x faster.
    """
    model = ap.wav2vec2
    n_layers_to_keep = config.W2V2_HIDDEN_LAYER + 1
    if len(model.encoder.layers) > n_layers_to_keep:
        model.encoder.layers = model.encoder.layers[:n_layers_to_keep]


def detector_logits_on_fakes(ap, logreg, fakes, paths, remembered):
    """Detector logit for each fake clip. The detector is frozen and the clips
    never change, so each clip's logit is computed once and remembered.

    fakes: (batch, 80000) -> logits: (batch, 1)
    """
    new_indices = []
    for i in range(len(paths)):
        if paths[i] not in remembered:
            new_indices.append(i)
    if len(new_indices) > 0:
        with torch.no_grad():
            features = ap.extract_features(fakes[new_indices])   # (n_new, frames, 1920)
            logits, _ = logreg(features.mean(dim=1))             # (n_new, 1)
        for row, i in enumerate(new_indices):
            remembered[paths[i]] = logits[row]

    batch_logits = []
    for path in paths:
        batch_logits.append(remembered[path])
    return torch.stack(batch_logits)  # (batch, 1)


def train(setup, *, fill="real", filelist=None, data_dir=None, epochs=1000, batch_size=15,
          lr=3e-5, out_dir=None, resume=False, workers=4):
    setup = config.get_setup(setup) if isinstance(setup, str) else setup
    check_fill(setup, fill)
    data_dir = Path(data_dir or setup.data_dir)
    out_dir = Path(out_dir or (config.RUNS_DIR / f"{setup.name}_{fill}fill"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "loss_terms.txt"

    accelerator = Accelerator()
    device = accelerator.device
    ap = AudioProcessor(device=device)
    keep_only_needed_wav2vec2_layers(ap)
    model = UNet().to(device)
    logreg = TorchLogReg(setup.detector_ckpt).to(device)
    loss_fn = LMACLoss(ap, logreg, fill=fill).to(device)

    opt_model = torch.optim.Adam(model.parameters(), lr=lr)

    state_path = out_dir / "last_state.pt"
    start_epoch = 0
    if resume and state_path.exists():
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state["model"])
        opt_model.load_state_dict(state["opt_model"])
        start_epoch = state["epoch"]

    # File list: explicit metadata (repo-relative paths), else every wav of the
    # setup under the data dir (setup.file_pattern, e.g. only the 1-2 kHz swaps).
    if filelist is not None:
        names = read_paths(filelist)
        root = config.PROJECT_ROOT
        # The metadata file must also follow the setup's pattern (no other bands sneak in).
        for name in names:
            if not fnmatch.fnmatch(Path(name).name, setup.file_pattern):
                raise ValueError(f"{name} in {filelist} does not match {setup.file_pattern!r}")
    else:
        names = []
        for wav in sorted(Path(data_dir).glob(setup.file_pattern)):
            names.append(wav.name)
        if len(names) == 0:  # wavs in sub-folders
            for wav in sorted(Path(data_dir).rglob(setup.file_pattern)):
                names.append(str(wav.relative_to(data_dir)))
        if len(names) == 0:
            raise FileNotFoundError(f"no files matching {setup.file_pattern!r} under {data_dir}")
        root = data_dir
    real_dir = setup.real_dir if fill == "real" else None
    reference = load_fill_reference(setup, fill, device)
    dataset = FakeAndRealDataset(names, ap, root=root, real_dir=real_dir)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fake_and_real,
        num_workers=workers, worker_init_fn=one_thread_per_worker, persistent_workers=workers > 0,
    )
    remembered_logits = {}

    model, opt_model, loader = accelerator.prepare(model, opt_model, loader)

    accelerator.print(
        f"Training '{setup.name}' ({fill} fill) on {len(dataset)} files, "
        f"{accelerator.num_processes} processes x batch {batch_size}, from epoch {start_epoch + 1} -> {out_dir}"
    )
    for epoch in range(start_epoch, epochs):
        model.train()
        sums = torch.zeros(4, device=device)  # total loss, l_in, l_out, L1
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}", ascii=True, dynamic_ncols=True,
                   disable=not accelerator.is_main_process)
        for fakes, reals, paths in bar:
            fakes = fakes.to(device)   # (batch, 80000)
            reals = reals.to(device)   # (batch, 80000)
            yhat_logits = detector_logits_on_fakes(ap, logreg, fakes, paths, remembered_logits)  # (batch, 1)
            with torch.no_grad():
                x_fake, _, _ = ap.compute_stft(fakes)   # (batch, 513, 313) complex
                x_real, _, _ = ap.compute_stft(reals)   # (batch, 513, 313) complex
                x_real = fill_spectrogram(fill, reference, x_fake, x_real)

            mask = model(x_fake.abs().unsqueeze(1))     # (batch, 1, 513, 313)
            loss_value, parts, weights = loss_fn.loss_function(
                mask, x_fake, x_real, torch.sigmoid(yhat_logits)
            )
            opt_model.zero_grad()
            accelerator.backward(loss_value)
            opt_model.step()

            sums += torch.cat([loss_value.detach().view(1), parts.detach()])
            bar.set_postfix(loss=f"{loss_value.item():.4f}")

        # average over the batches of this GPU, then over all GPUs
        means = accelerator.reduce(sums / len(loader), reduction="mean")
        avg_loss, l_in, l_out, l1 = means.tolist()
        if accelerator.is_main_process:
            ckpt = out_dir / f"UNet_MP_epoch_{epoch + 1}_loss_{avg_loss:.4f}.pth"
            accelerator.save(accelerator.unwrap_model(model).state_dict(), ckpt)
            accelerator.save({
                "epoch": epoch + 1,
                "model": accelerator.unwrap_model(model).state_dict(),
                "opt_model": opt_model.state_dict(),
            }, state_path)
            with open(log_path, "a") as f:
                f.write(
                    f"Epoch {epoch + 1}: l_in={l_in:.4f}, l_out={l_out:.4f}, L1={l1:.4f}, "
                    f"weights={weights.cpu().numpy()}\n"
                )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setup", choices=config.SETUP_NAMES, required=True)
    p.add_argument("--fill", choices=FILLS, default="real",
                   help="real = fill masked-out bins from the bona fide clip; avg = from the setup's "
                        "average spectrogram (compute_avg_spectrogram.py); avgreal = from the average bona fide "
                        "spectrogram (compute_avg_spectrogram.py --source real); optimized = from the fill "
                        "optimised to be neutral for the setup's detector (learn_neutral_fill.py); "
                        "zero = original loss")
    p.add_argument("--filelist", default=None, help="metadata file; default = all wavs of the setup in the data dir")
    p.add_argument("--data-dir", default=None, help="override the setup's data directory")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--out-dir", default=None, help="default: runs/<setup>_<fill>fill")
    p.add_argument("--resume", action="store_true", help="continue from <out-dir>/last_state.pt if present")
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers per GPU process")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        args.setup, fill=args.fill, filelist=args.filelist, data_dir=args.data_dir, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, out_dir=args.out_dir,
        resume=args.resume, workers=args.workers,
    )
