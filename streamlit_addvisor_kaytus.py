import os

os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
from addvisor import ADDvisor
from audioprocessor import AudioProcessor
from classifier_embedder import TorchLogReg, TorchScaler

from tqdm import tqdm
import io
from torch.utils.data import Dataset, DataLoader
from pyngrok import ngrok
from accelerate import load_checkpoint_and_dispatch

ngrok.kill()

st.set_page_config(layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

audio_processor = AudioProcessor()
model = ADDvisor().to(device)
torch_log_reg = TorchLogReg().to(device)
torch_scaler = TorchScaler().to(device)

# checkpoint_path = r'C:\Users\david\PycharmProjects\David2\model\addvisor_epoch_92_loss_0.3716.pth'
checkpoint_path = "/mnt/QNAP/comdav/addvisor_savedV7/addvisor_epoch_92_loss_0.3716.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
if any(k.startswith("module.") for k in checkpoint.keys()):
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
model.load_state_dict(checkpoint)

# model = load_checkpoint_and_dispatch(
#     model, checkpoint=checkpoint_path, device_map="auto", no_split_module_classes=['Block']
# )
model.eval()
torch_log_reg.eval()


def find_all_wav_files2(root_dir, max_files=None, systems=False):
    audio_files = []
    #    audio_files_systems = []

    #    if systems:
    #        for dirpath, subdirs, files in os.walk(root_dir):
    #            for
    #    else:
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(dirpath, file))
                if max_files and len(audio_files) >= max_files:
                    return audio_files
    return audio_files


@st.cache_data
def plot_spectrogram(spec, title):
    fig, ax = plt.subplots()
    ax.imshow(np.log1p(spec), aspect="auto", origin="lower")
    ax.set_title(title)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


@st.cache_data
def plot_spectrogram_logmag(spec, title):
    fig, ax = plt.subplots()
    ax.imshow(spec, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


# @st.cache_data
# def plot_mask_scatter(mask_tensor, title="Mask Scatter Plot"):
#     flat_mask = mask_tensor.flatten()
#     x = np.arange(len(flat_mask))
#     fig, ax = plt.subplots()
#     ax.scatter(x, flat_mask, s=5, alpha=0.6, color='blue')
#     ax.axhline(0, linestyle='--', color='red', linewidth=1)
#     ax.set_title(title)
#     ax.set_xlabel("Index")
#     ax.set_ylabel("Value")
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png")
#     plt.close(fig)
#     buf.seek(0)
#     return buf


class AudioDataset(Dataset):
    def __init__(self, directory1, directory2, audio_processor, device):
        self.file_paths = find_all_wav_files2(
            directory1, max_files=30
        ) + find_all_wav_files2(directory2, max_files=30)
        self.audio_processor = audio_processor
        self.device = device

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        print(audio_path)
        waveform = self.audio_processor.load_audio(audio_path)[0]
        return waveform.to(self.device), audio_path


@st.cache_resource(show_spinner=True)
def run_addvisor_batched(dir_path1, dir_path2):
    dataset = AudioDataset(dir_path1, dir_path2, audio_processor, device)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False)
    results = []

    for waveforms, paths in tqdm(data_loader):
        feats = audio_processor.extract_features(waveforms)
        feats_mean = torch.mean(feats, dim=1)
        yhat1_logits, yhat1_probs = torch_log_reg(feats_mean)

        mask = model(feats)
        #        print("mask stats: min =", mask.min().item(), " max =", mask.max().item(), " mean =", mask.mean().item())
        #        mean_mask = mask.mean().item()
        #        mask = (mask > mean_mask).float()

        _, magnitude, phase = audio_processor.compute_stft(waveforms)
        Tmax = mask.shape[1]
        log_mag = torch.log1p(magnitude[:, :Tmax, :]).to(device)
        phase = phase[:, :Tmax, :].to(device)

        masked_log_mag_for_vis = mask * log_mag
        compl_masked_log_mag_for_vis = (1 - mask) * log_mag

        relevant_mask_stft = torch.expm1(mask * log_mag)
        irrelevant_mask_stft = torch.expm1((1 - mask) * log_mag)
        relevant_mask = relevant_mask_stft * torch.exp(1j * phase)
        irrelevant_mask = irrelevant_mask_stft * torch.exp(1j * phase)
        istft_relevant_mask = audio_processor.compute_invert_stft(relevant_mask)
        istft_irrelevant_mask = audio_processor.compute_invert_stft(irrelevant_mask)
        istft_feats = audio_processor.extract_features(istft_relevant_mask)
        istft_irrelevant_feats = audio_processor.extract_features(istft_irrelevant_mask)
        istft_feats_mean = torch.mean(istft_feats, dim=1)
        istft_irrelevant_feats_mean = torch.mean(istft_irrelevant_feats, dim=1)
        _, yhat2_probs = torch_log_reg(istft_feats_mean)
        _, yhat3_probs = torch_log_reg(istft_irrelevant_feats_mean)

        for i in range(waveforms.size(0)):
            results.append(
                {
                    "filename": os.path.basename(paths[i]),
                    "original_audio": waveforms[i].cpu().numpy(),
                    "reconstructed_audio": istft_relevant_mask[i]
                    .detach()
                    .cpu()
                    .numpy(),
                    "spectrogram_img": plot_spectrogram(
                        magnitude[i].cpu().numpy(), "Spectrogram"
                    ),
                    "mask_img": plot_spectrogram(
                        mask[i].detach().cpu().numpy(), "Mask"
                    ),
                    "mask_img_compl": plot_spectrogram(
                        1 - mask[i].detach().cpu().numpy(), "1 - Mask"
                    ),
                    "masked_spectrogram_img": plot_spectrogram_logmag(
                        masked_log_mag_for_vis[i].detach().cpu().numpy(),
                        "Spectrogram x Mask",
                    ),
                    "compl_masked_spectrogram_img": plot_spectrogram_logmag(
                        compl_masked_log_mag_for_vis[i].detach().cpu().numpy(),
                        "Spectrogram x (1 - Mask)",
                    ),
                    "pred_original": yhat1_probs[i].cpu().detach().numpy(),
                    "pred_reconstructed_mask": yhat2_probs[i].cpu().detach().numpy(),
                    "pred_reconstructed_1-mask": yhat3_probs[i].cpu().detach().numpy(),
                    # "mask_scatter": plot_mask_scatter(mask[i].detach().cpu().numpy(), "Mask Scatter")
                }
            )

    return results


# DIR_PATH1 = r"C:\Machine_Learning_Data\Deepfake_datasets\mlaad_v5"
# DIR_PATH2 = r"C:\Machine_Learning_Data\Deepfake_datasets\m-ailabs"
DIR_PATH1 = "/mnt/QNAP/comdav/MLAAD_v5/"
DIR_PATH2 = "/mnt/QNAP/comdav/m-ailabs/"
results = run_addvisor_batched(DIR_PATH1, DIR_PATH2)

st.title("quality visualization of explainability")

for item in results:
    st.subheader(item["filename"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**original audio**")
        st.audio(item["original_audio"], format="audio/wav", sample_rate=16000)
    with col2:
        st.markdown("**reconstructed audio**")
        st.audio(item["reconstructed_audio"], format="audio/wav", sample_rate=16000)

    img_col1, img_col2, img_col3, img_col4, img_col5, img_col6 = st.columns(6)
    with img_col1:
        st.image(
            item["spectrogram_img"], caption="spectrogram", use_container_width=True
        )
    with img_col2:
        st.image(item["mask_img"], caption="mask", use_container_width=True)
    with img_col4:
        st.image(
            item["masked_spectrogram_img"],
            caption="spectrogram x mask",
            use_container_width=True,
        )
    with img_col5:
        st.image(item["mask_img_compl"], caption="1 - mask ", use_container_width=True)
    with img_col6:
        st.image(
            item["compl_masked_spectrogram_img"],
            caption="spectrogram x (1-mask)",
            use_container_width=True,
        )

    st.markdown("**predictions**")
    st.write("on original audio: ", item["pred_original"])
    st.write("on reconstructed: ", item["pred_reconstructed_mask"])
    st.write("on (1- mask) * audio: ", item["pred_reconstructed_1-mask"])
    st.markdown("---")


############## TODO
## plotez features w2v si inainte si dupa masca (line plot??? )
## scot audio reals dupa masca si le adun la audio fakes initiale... o sa schimbe decizia deepfake detectorului?
## mai verific codul
## schimb culorile din streamlit sa fie mai vizibile
## pastrez doar 5% cele mai mai valori, restul 0 --> cat de tare schimba metricile de faithfulness si fidelity??
