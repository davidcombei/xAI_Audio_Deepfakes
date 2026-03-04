import click

import streamlit as st

import torch
from torch.utils.data import DataLoader

from addvisor import UNet
from audioprocessor import AudioProcessor


DEVICE = "cuda"


def extract_wavs(path):
    fake_paths = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            path = line.rsplit(",", 1)[0].strip()
            fake_paths.append(path)
    return fake_paths


def collate_fn(audio_processor, batch):
    waveforms, audio_paths = zip(*batch)
    waveforms = torch.stack(waveforms, dim=0)
    _, magnitude, phase = audio_processor.compute_stft(waveforms)
    features = audio_processor.extract_features(waveforms)
    return features, magnitude, phase, audio_paths


class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path_metadata,
        audio_processor,
        device,
    ):

        self.file_paths = extract_wavs(path_metadata)
        self.audio_processor = audio_processor
        self.device = device

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        waveform, _ = self.audio_processor.load_audio(path)
        return waveform.to(self.device), path


PATHS_METADATA = {
    "realistic": "metadata/mlaad_selection_metadata.txt",
    "freq-swapped": "metadata/ljspeech_manipulated_metadata.txt",
    # "time-swapped": "data/metadata/time-swapped.csv",
}


def build_dataloader(setup):
    batch_size = 4
    audio_processor = AudioProcessor()
    collate_fn_with_processor = lambda batch: collate_fn(audio_processor, batch)
    dataset = AudioDataset(PATHS_METADATA[setup], audio_processor, DEVICE)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_with_processor)
    return loader


def build_decoder(setup):
    CHECKPOINTS = {
        "realistic": "ckpts/ckpts_realisticDeepfakes/UNet_MP_epoch_41_loss_0.3316.pth",
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


@click.command()
@click.option("--setup", type=click.Choice(["realistic", "time-swapped", "freq-swapped"], case_sensitive=False), default="realistic", help="Choose the dataset setup to use.")
def main(setup):
    loader = build_dataloader(setup)
    decoder = build_decoder(setup)

    for batch in loader:
        features, magnitude, phase, audio_paths = batch
        with torch.no_grad():
            pred = decoder(magnitude.unsqueeze(1))
        # print("Features shape:", features.shape)
        # print("Magnitude shape:", magnitude.shape)
        # print("Phase shape:", phase.shape)
        # print("Audio paths:", audio_paths)
        # import pdb; pdb.set_trace()



if __name__ == "__main__":
    main()