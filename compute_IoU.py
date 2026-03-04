import click
import os
import random

import joblib
import streamlit as st
import torch

from captum.attr import Saliency, InputXGradient, IntegratedGradients, GradientShap
from sklearn.metrics import roc_auc_score
from streamlit.runtime.scriptrunner import get_script_run_ctx
from torch.utils.data import DataLoader
from torch import nn

from addvisor import UNet
from audioprocessor import AudioProcessor
from classifier_embedder import TorchLogReg
from skimage.segmentation import felzenszwalb

import numpy as np


DEVICE = "cuda"
random.seed(42)


def collate_fn(audio_processor, batch):
    waveforms, audio_paths = zip(*batch)
    waveforms = torch.stack(waveforms, dim=0)
    _, magnitude, phase = audio_processor.compute_stft(waveforms)
    features = audio_processor.extract_features(waveforms)
    return features, magnitude, phase, audio_paths


class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        folder,
        audio_processor,
        device,
    ):
        self.file_paths = os.listdir(folder)
        self.file_paths = [
            os.path.join(folder, file)
            for file in self.file_paths
            if file.endswith(".wav")
        ]
        # TODO Fix subset
        self.file_paths = random.sample(self.file_paths, 100)
        self.audio_processor = audio_processor
        self.device = device

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        waveform, _ = self.audio_processor.load_audio(path)
        return waveform.to(self.device), path


FOLDERS = {
    "freq-swapped": "LJSpeech_freq_swap",
    "time-swapped": "LJSpeech_time_swap",
}


def build_dataloader(setup):
    batch_size = 4
    audio_processor = AudioProcessor()
    collate_fn_with_processor = lambda batch: collate_fn(audio_processor, batch)
    dataset = AudioDataset(FOLDERS[setup], audio_processor, DEVICE)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_with_processor,
    )
    return loader


def build_decoder(setup):
    CHECKPOINTS = {
        # "realistic": "ckpts/ckpts_realisticDeepfakes/UNet_MP_epoch_41_loss_0.3316.pth",
        "freq-swapped": "ckpts/ckpts_freqSwapDeepfakes/UNet_MP_epoch_74_loss_0.0182.pth",
        "time-swapped": "ckpts/ckpts_timeSwapDeepfakes/UNet_MP_epoch_151_loss_0.5357.pth",
    }
    model = UNet().to(DEVICE)
    checkpoint_path = CHECKPOINTS[setup]
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    if any(k.startswith("module.") for k in checkpoint.keys()):
        new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        checkpoint = new_state_dict
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def get_true_freq(audio_path, shape):
    F, T = shape

    # Get the frequency range from the file name
    file_name = os.path.basename(audio_path)
    file_name, _ = os.path.splitext(file_name)

    *_, f_low_f_high = file_name.split("_")
    f_low, f_high = f_low_f_high.split("-")
    f_low = int(f_low)
    f_high = int(f_high)

    f_max = 16_000 / 2
    idx_low = int((f_low / f_max) * F)
    idx_high = int((f_high / f_max) * F)

    freq_mask = torch.zeros((F, T))
    freq_mask[idx_low:idx_high] = 1.0

    return freq_mask


class TorchLogReg(nn.Module):
    def __init__(self, classifier_path):
        super(TorchLogReg, self).__init__()
        classifier = joblib.load(classifier_path)

        self.linear = nn.Linear(1920, 1)
        self.linear.weight = nn.Parameter(
            torch.tensor(classifier.coef_, dtype=torch.float32),
            requires_grad=False,
        )
        self.linear.bias = nn.Parameter(
            torch.tensor(classifier.intercept_, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, x):
        logits = self.linear(x)
        probs = torch.sigmoid(logits)
        return logits, probs


class Wav2vec2LogReg(nn.Module):
    def __init__(self, audioprocessor, logReg):
        super().__init__()
        self.ap = audioprocessor
        self.logReg = logReg

    def forward(self, magnitude, phase):
        spec = magnitude * torch.exp(1j * phase)
        audio = self.ap.compute_invert_stft(spec)
        feats = self.ap.extract_features(audio)
        feats = feats.mean(dim=1)  # Average over time dimension
        logits, _ = self.logReg(feats)
        return logits


def build_explainer(setup, explanation_method):
    if explanation_method == "optimized":
        decoder = build_decoder(setup)

        def explainer(magnitude, phase):
            with torch.no_grad():
                pred = decoder(magnitude.unsqueeze(1))
                pred = pred.squeeze(1)
            return pred

    elif explanation_method == "random":

        def segment1(mag):
            segments = felzenszwalb(mag, scale=15, sigma=0.5)
            random_mask = np.zeros_like(mag)
            for segment_id in np.unique(segments):
                random_value = np.random.uniform(0, 1)
                random_mask[segments == segment_id] = random_value
            return random_mask

        def explainer(magnitude, phase):
            magnitude_np = magnitude.cpu().numpy()
            expl = [segment1(mag) for mag in magnitude_np]
            expl = torch.tensor(expl)
            expl = expl.to(DEVICE)
            return expl

    elif explanation_method in ["saliency", "input-x-gradient"]:
        CLFS = {
            "freq-swapped": "ckpts/logReg_ckpts/logReg_vocoded_anyband_controlled.joblib",
            "time-swapped": "ckpts/logReg_ckpts/logReg_vocoded_TIMESWAP.joblib",
        }
        classifier_path = CLFS[setup]
        torch_log_reg = TorchLogReg(classifier_path)

        model = Wav2vec2LogReg(AudioProcessor(), torch_log_reg)
        model.to(DEVICE)
        model.eval()

        CAPTUM = {
            "saliency": lambda model: Saliency(model),
            "input-x-gradient": lambda model: InputXGradient(model),
        }

        def explainer(magnitude, phase):
            magnitude = magnitude.requires_grad_(True)
            explainer = CAPTUM[explanation_method](model)
            attr = explainer.attribute(magnitude, additional_forward_args=(phase,))
            return attr

    else:
        raise ValueError(f"Unsupported explanation method: {explanation_method}")

    return explainer


def get_true_time(audio_path, shape):
    F, T = shape
    # T corresonds to 5 seconds
    # The mask is from 3 to 5 seconds
    start_frame = int(3 / 5 * T)
    time_mask = torch.zeros((F, T))
    time_mask[:, start_frame:] = 1.0
    return time_mask


GET_TRUE = {
    "freq-swapped": get_true_freq,
    "time-swapped": get_true_time,
}


def min_max_scaling(x):
    x_min = x.min()
    x_max = x.max()
    if x_max - x_min == 0:
        return torch.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


@click.command()
@click.option(
    "-s",
    "--setup",
    type=click.Choice(["time-swapped", "freq-swapped"]),
    default="realistic",
    help="Choose the dataset setup to use.",
)
@click.option(
    "-e",
    "--explanation",
    "explanation_method",
    type=click.Choice(["random", "saliency", "input-x-gradient", "optimized"]),
    help="Choose the explanation method to use.",
)
def main(setup, explanation_method):
    loader = build_dataloader(setup)
    explainer = build_explainer(setup, explanation_method)
    aucs = []

    is_streamlit = get_script_run_ctx() is not None

    for batch in loader:
        features, magnitude, phase, audio_paths = batch
        _, F, T = magnitude.shape

        with torch.no_grad():
            # TODO Cache the predictions for optimized decoder to avoid redundant computation
            pred = explainer(magnitude, phase)
            true = [GET_TRUE[setup](audio_path, (F, T)) for audio_path in audio_paths]

        for i in range(pred.shape[0]):
            pred_img = pred[i].cpu().numpy()
            pred_img = min_max_scaling(pred_img)
            true_img = true[i].cpu().numpy()

            auc = roc_auc_score(true_img.flatten(), pred_img.flatten())
            aucs.append(auc)
            print(auc)

            if is_streamlit:
                st.image(
                    pred_img,
                    caption=f"Predicted Mask - AUC: {auc:.4f}",
                    use_column_width=True,
                )
                st.image(true_img, caption="True Mask", use_column_width=True)

        if is_streamlit:
            import pdb

            pdb.set_trace()

    auc_mean = sum(aucs) / len(aucs)
    print(f"Average AUC: {auc_mean:.4f}")


if __name__ == "__main__":
    main()
