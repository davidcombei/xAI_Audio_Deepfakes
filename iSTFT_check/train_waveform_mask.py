"""Train FrameMaskNet, the waveform mask predictor (the waveform version of train_mask.py).

Same recipe as the main pipeline's U-Net: all wavs of the setup's folder, the
LMAC loss with learned weights, Adam with lr 3e-5 (model) and 1e-4 (loss
weights), one checkpoint per epoch. The difference: the mask multiplies the
waveform directly, so there is no STFT and no iSTFT.

Run (on 8 GPUs; --gpu_ids is needed because this machine's default accelerate
config pins its own list of GPUs):
  accelerate launch --multi_gpu --gpu_ids 0,1,2,3,4,5,6,7 --num_processes 8 \\
      iSTFT_check/train_waveform_mask.py --setup time-swapped --epochs 400 --batch-size 4 --resume

Output (runs/waveform_mask/<setup>/):
  FrameMaskNet_epoch_<N>_loss_<L>.pth   the model after every epoch
  last_state.pt                         everything needed to continue with --resume
  loss_terms.txt                        the loss terms of every epoch
"""

# The main ADDvisor modules (config, audio, detector, ...) live one folder up.
# Add that folder to Python's search path so we can import them.
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from common import load_audio_processor, use_stable_attention
from data import AudioDataset, read_paths
from models import TorchLogReg
from waveform_model import FrameMaskNet, WaveformLMACLoss


def main():
    args = parse_args()
    train(args.setup, filelist=args.filelist, data_dir=args.data_dir, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, w_lr=args.w_lr, out_dir=args.out_dir, resume=args.resume)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setup", choices=config.SETUP_NAMES, default="time-swapped")
    parser.add_argument("--filelist", default=None, help="text file with wav names; default = all wavs in the data dir")
    parser.add_argument("--data-dir", default=None, help="use another folder of wavs than the setup's")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=15, help="clips per GPU")
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate of the model")
    parser.add_argument("--w-lr", type=float, default=1e-4, help="learning rate of the loss weights")
    parser.add_argument("--out-dir", default=None, help="default: runs/waveform_mask/<setup>")
    parser.add_argument("--resume", action="store_true", help="continue from <out-dir>/last_state.pt if it exists")
    return parser.parse_args()


def train(setup_name, filelist=None, data_dir=None, epochs=1000, batch_size=15,
          lr=3e-5, w_lr=1e-4, out_dir=None, resume=False):
    setup = config.get_setup(setup_name)
    use_stable_attention()

    # 1. Folders.
    if data_dir is None:
        data_dir = setup.data_dir
    data_dir = Path(data_dir)
    if out_dir is None:
        out_dir = config.RUNS_DIR / "waveform_mask" / setup.name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Models. `accelerator` takes care of using several GPUs.
    accelerator = Accelerator()
    device = accelerator.device
    audio_processor = load_audio_processor(device)
    model = FrameMaskNet().to(device)
    detector_head = TorchLogReg(setup.detector_ckpt).to(device)      # logistic regression of the detector
    loss_function = WaveformLMACLoss(audio_processor, detector_head).to(device)

    # 3. Optimizers: one for the model, one for the three loss weights.
    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weight_optimizer = torch.optim.Adam([loss_function.w_raw], lr=w_lr)

    # 4. Continue an earlier run? (This also works with a different number of GPUs;
    #    keep batch_size x number of GPUs about the same.)
    state_file = out_dir / "last_state.pt"
    start_epoch = 0
    if resume and state_file.exists():
        state = torch.load(state_file, map_location=device)
        model.load_state_dict(state["model"])
        model_optimizer.load_state_dict(state["opt_model"])
        weight_optimizer.load_state_dict(state["opt_w"])
        with torch.no_grad():
            loss_function.w_raw.copy_(state["w_raw"])
        start_epoch = state["epoch"]

    # 5. Data.
    if filelist is not None:
        names = read_paths(filelist)
    else:
        names = []
        for wav_file in sorted(data_dir.glob("*.wav")):
            names.append(wav_file.name)
    dataset = AudioDataset(names, audio_processor, root=data_dir, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model, model_optimizer, weight_optimizer, loader = accelerator.prepare(
        model, model_optimizer, weight_optimizer, loader)
    accelerator.print(f"Training FrameMaskNet on {setup.name}: {len(dataset)} files, "
                      f"{accelerator.num_processes} GPU(s) x batch {batch_size}, from epoch {start_epoch + 1}")

    # 6. Training.
    for epoch in range(start_epoch, epochs):
        model.train()
        sum_loss, sum_keep, sum_remove, sum_size = 0.0, 0.0, 0.0, 0.0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}", ascii=True,
                        disable=not accelerator.is_main_process)
        for waveform, _ in progress:
            # The detector's original answer on these clips: the target of the loss.
            with torch.no_grad():
                original_prob = torch.sigmoid(loss_function.detector_logit(waveform))     # (batch, 1)

            frame_mask = model(waveform)                                                     # (batch, 248)
            loss, terms, weights = loss_function.loss_function(frame_mask, waveform, original_prob)

            model_optimizer.zero_grad()
            weight_optimizer.zero_grad()
            accelerator.backward(loss)
            # The loss weights are not wrapped by `accelerator`, so we average their
            # gradient over the GPUs ourselves (otherwise every GPU learns different weights).
            if accelerator.num_processes > 1:
                loss_function.w_raw.grad = accelerator.reduce(loss_function.w_raw.grad, reduction="mean")
            model_optimizer.step()
            weight_optimizer.step()

            sum_loss = sum_loss + loss.item()
            sum_keep = sum_keep + terms[0].item()
            sum_remove = sum_remove + terms[1].item()
            sum_size = sum_size + terms[2].item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        # Average over the batches of this GPU, then over all GPUs.
        n_batches = len(loader)
        averages = torch.tensor([sum_loss, sum_keep, sum_remove, sum_size], device=device) / n_batches
        averages = accelerator.reduce(averages, reduction="mean")
        avg_loss = averages[0].item()
        avg_keep = averages[1].item()
        avg_remove = averages[2].item()
        avg_size = averages[3].item()

        # 7. Save (only one GPU writes files).
        if accelerator.is_main_process:
            model_weights = accelerator.unwrap_model(model).state_dict()
            accelerator.save(model_weights, out_dir / f"FrameMaskNet_epoch_{epoch + 1}_loss_{avg_loss:.4f}.pth")
            accelerator.save({
                "epoch": epoch + 1,
                "model": model_weights,
                "opt_model": model_optimizer.state_dict(),
                "opt_w": weight_optimizer.state_dict(),
                "w_raw": loss_function.w_raw.detach(),
            }, state_file)
            with open(out_dir / "loss_terms.txt", "a") as log:
                log.write(f"Epoch {epoch + 1}: l_in={avg_keep:.4f}, l_out={avg_remove:.4f}, L1={avg_size:.4f}, "
                          f"weights={weights.detach().cpu().numpy()}\n")


if __name__ == "__main__":
    main()
