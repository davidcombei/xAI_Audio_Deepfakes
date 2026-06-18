from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from audio import AudioProcessor
from data import AudioDataset, collate_with_logits, find_wavs, read_paths
from losses import LMACLoss
from models import UNet, TorchLogReg

# The loss backprops through the frozen wav2vec2 via ISTFT features; the math
# SDP kernel keeps that attention backward numerically stable.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def train(setup, *, filelist=None, data_dir=None, epochs=1000, batch_size=15,
          lr=3e-5, w_lr=1e-4, out_dir=None):
    setup = config.get_setup(setup) if isinstance(setup, str) else setup
    data_dir = Path(data_dir or setup.data_dir)
    out_dir = Path(out_dir or (config.RUNS_DIR / setup.name))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "loss_terms.txt"

    accelerator = Accelerator()
    device = accelerator.device
    ap = AudioProcessor(device=device)
    model = UNet().to(device)
    logreg = TorchLogReg(setup.detector_ckpt).to(device)
    loss_fn = LMACLoss(ap, logreg).to(device)

    opt_model = torch.optim.Adam(model.parameters(), lr=lr)
    opt_w = torch.optim.Adam([loss_fn.w_raw], lr=w_lr)

    # File list: explicit metadata, else every wav under the data dir.
    if filelist is not None:
        names = read_paths(filelist)
    else:
        names = sorted(p.name for p in Path(data_dir).glob("*.wav")) or find_wavs(data_dir)
    dataset = AudioDataset(names, ap, root=data_dir, device=device)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=partial(collate_with_logits, audio_processor=ap, logreg=logreg),
    )

    model, opt_model, opt_w, loader = accelerator.prepare(model, opt_model, opt_w, loader)

    print(f"Training '{setup.name}' on {len(dataset)} files -> {out_dir}")
    for epoch in range(epochs):
        totals = {"loss": 0.0, "l_in": 0.0, "l_out": 0.0, "l1": 0.0}
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}", ascii=True, dynamic_ncols=True)
        for features, magnitude, phase, yhat_logits in bar:
            mask = model(magnitude.unsqueeze(1))
            loss_value, parts, weights = loss_fn.loss_function(
                mask, magnitude, phase, torch.sigmoid(yhat_logits)
            )
            opt_model.zero_grad()
            opt_w.zero_grad()
            accelerator.backward(loss_value)
            opt_model.step()
            opt_w.step()

            totals["loss"] += loss_value.item()
            totals["l_in"] += parts[0].item()
            totals["l_out"] += parts[1].item()
            totals["l1"] += parts[2].item()
            bar.set_postfix(loss=f"{loss_value.item():.4f}")

        n = len(loader)
        avg_loss = totals["loss"] / n
        ckpt = out_dir / f"UNet_MP_epoch_{epoch + 1}_loss_{avg_loss:.4f}.pth"
        accelerator.save(accelerator.unwrap_model(model).state_dict(), ckpt)
        with open(log_path, "a") as f:
            f.write(
                f"Epoch {epoch + 1}: l_in={totals['l_in'] / n:.4f}, "
                f"l_out={totals['l_out'] / n:.4f}, L1={totals['l1'] / n:.4f}, "
                f"weights={weights.detach().cpu().numpy()}\n"
            )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setup", choices=config.SETUP_NAMES, required=True)
    p.add_argument("--filelist", default=None, help="metadata file; default = all wavs in the data dir")
    p.add_argument("--data-dir", default=None, help="override the setup's data directory")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--w-lr", type=float, default=1e-4)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        args.setup, filelist=args.filelist, data_dir=args.data_dir, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, w_lr=args.w_lr, out_dir=args.out_dir,
    )
