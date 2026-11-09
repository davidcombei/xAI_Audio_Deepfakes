"""Train the ADDvisor U-Net mask predictor against the simple detector.

Same U-Net, same input (the fake clip's 512 x 248 STFT magnitude), same loss
terms, learnable loss weights and optimizers (Adam, lr 3e-5 / 1e-4) as the
repo's ``train_mask.py``. The only change: the detector is the mel logistic
regression of ``mel_detector.py`` instead of wav2vec2 + logistic regression.

--fill zero  removed bins are set to zero (the original ADDvisor recipe)
--fill real  removed bins are taken from the bona fide clip (new_masking/)

Run from the repo root, e.g. on GPUs 0-1:

    accelerate launch --multi_gpu --gpu_ids 0,1 --num_processes 2 \\
        simple_classifier/train_mask.py --setup freq-swapped --fill zero --epochs 60 --resume
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root: config.py, audio.py, new_masking/

import argparse

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from audio import AudioProcessor
from common import (
    RUNS_DIR, SETUP_NAMES, FakeAndRealDataset, collate_fake_and_real, complex_stft,
    detector_logits, detector_path, list_files, one_thread_per_worker,
)
from lmac_loss import MelLMACLoss
from mel_detector import MelLogRegDetector
from models import UNet


def train(setup_name, fill="zero", epochs=60, batch_size=15, lr=3e-5, w_lr=1e-4,
          out_dir=None, resume=False, workers=6, save_every=10):
    setup = config.get_setup(setup_name)
    if out_dir is None:
        out_dir = RUNS_DIR / f"{setup.name}_{fill}fill"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "loss_terms.txt"

    accelerator = Accelerator()
    device = accelerator.device
    ap = AudioProcessor(device=device)
    model = UNet().to(device)
    detector = MelLogRegDetector(detector_path(setup.name)).to(device)
    loss_fn = MelLMACLoss(ap, detector, fill=fill).to(device)

    opt_model = torch.optim.Adam(model.parameters(), lr=lr)
    opt_w = torch.optim.Adam([loss_fn.w_raw], lr=w_lr)

    # Resume from the last finished epoch (also works with a different number of GPUs).
    state_path = out_dir / "last_state.pt"
    start_epoch = 0
    if resume and state_path.exists():
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state["model"])
        opt_model.load_state_dict(state["opt_model"])
        opt_w.load_state_dict(state["opt_w"])
        with torch.no_grad():
            loss_fn.w_raw.copy_(state["w_raw"])
        start_epoch = state["epoch"]

    dataset = FakeAndRealDataset(list_files(setup), ap)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fake_and_real,
        num_workers=workers, worker_init_fn=one_thread_per_worker, persistent_workers=workers > 0,
    )

    model, opt_model, opt_w, loader = accelerator.prepare(model, opt_model, opt_w, loader)

    accelerator.print(
        f"Training UNet ({fill} fill, mel logReg detector) on '{setup.name}' ({len(dataset)} files, "
        f"{accelerator.num_processes} processes x batch {batch_size}) from epoch {start_epoch + 1} -> {out_dir}"
    )
    for epoch in range(start_epoch, epochs):
        model.train()
        sums = torch.zeros(4, device=device)  # total loss, l_in, l_out, L1
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}", ascii=True, dynamic_ncols=True,
                   disable=not accelerator.is_main_process, mininterval=30)
        for fakes, reals, paths in bar:
            with torch.no_grad():
                x_fake = complex_stft(ap, fakes.to(device))   # (batch, 512, 248)
                x_real = complex_stft(ap, reals.to(device))   # (batch, 512, 248)
                yhat_logits = detector_logits(ap, detector, x_fake)  # (batch,) detector's verdict on the fake

            mask = model(x_fake.abs().unsqueeze(1))  # (batch, 1, 512, 248)
            loss_value, parts, weights = loss_fn.loss_function(mask, x_fake, x_real, torch.sigmoid(yhat_logits))
            opt_model.zero_grad()
            opt_w.zero_grad()
            accelerator.backward(loss_value)
            # loss_fn is not wrapped by accelerate, so average the gradient of the
            # loss weights over the GPUs by hand (every GPU must keep the same weights).
            if accelerator.num_processes > 1:
                loss_fn.w_raw.grad = accelerator.reduce(loss_fn.w_raw.grad, reduction="mean")
            opt_model.step()
            opt_w.step()

            sums += torch.cat([loss_value.detach().view(1), parts.detach()])
            bar.set_postfix(loss=f"{loss_value.item():.4f}", refresh=False)

        # average over the batches of this GPU, then over all GPUs
        means = accelerator.reduce(sums / len(loader), reduction="mean")
        avg_loss, l_in, l_out, l1 = means.tolist()
        if accelerator.is_main_process:
            is_last_epoch = epoch + 1 == epochs
            if (epoch + 1) % save_every == 0 or is_last_epoch:
                ckpt = out_dir / f"UNet_MP_epoch_{epoch + 1}_loss_{avg_loss:.4f}.pth"
                accelerator.save(accelerator.unwrap_model(model).state_dict(), ckpt)
            accelerator.save({
                "epoch": epoch + 1,
                "model": accelerator.unwrap_model(model).state_dict(),
                "opt_model": opt_model.state_dict(),
                "opt_w": opt_w.state_dict(),
                "w_raw": loss_fn.w_raw.detach(),
            }, state_path)
            with open(log_path, "a") as f:
                f.write(
                    f"Epoch {epoch + 1}: l_in={l_in:.4f}, l_out={l_out:.4f}, L1={l1:.4f}, "
                    f"weights={weights.detach().cpu().numpy()}\n"
                )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setup", choices=SETUP_NAMES, required=True)
    p.add_argument("--fill", default="zero", choices=["zero", "real"])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--w-lr", type=float, default=1e-4)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--resume", action="store_true", help="continue from <out-dir>/last_state.pt if present")
    p.add_argument("--workers", type=int, default=6, help="DataLoader workers per GPU process")
    p.add_argument("--save-every", type=int, default=10, help="keep a U-Net checkpoint every N epochs (and the last)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.setup, fill=args.fill, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
          w_lr=args.w_lr, out_dir=args.out_dir, resume=args.resume, workers=args.workers,
          save_every=args.save_every)
