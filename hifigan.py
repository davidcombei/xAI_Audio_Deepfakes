# import torchaudio
# from speechbrain.inference.vocoders import HIFIGAN
# from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
# import os
# import torch
# from tqdm import tqdm
# import torchaudio.transforms as T
# import librosa
# # Load a pretrained HIFIGAN Vocoder
# hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="pretrained_models/hifigan_16k")

# #signal, rate = torchaudio.load('speechbrain/tts-hifigan-libritts-22050H/example_22kHz.wav')
# file_names = []
# with open("metadata/ljspeech_manipulated_metadata.txt", "r") as f:
#     for line in f:
#         file_names.append(line.strip())
# file_names = file_names[:5000]
# #file_names = file_names[:3]
# print('nr of files:', len(file_names))

# for file_name in tqdm(file_names):
#    # with open("ljspeech_manipulated_metadata.txt", "w") as f:
#    #     f.write(file_name+"\n")
#     file_name = file_name.split(',')[0]
#     print('##processing file:', file_name)
#     signal, rate = librosa.load(os.path.join("/mnt/QNAP/comdav/DATA/DATA/LJSpeech/wavs/", file_name), sr=16000)
#     signal = torch.tensor(signal)
#     if signal.ndim > 1:
#         signal = signal[0]
#     spectrogram, _ = mel_spectogram(
#         audio=signal,
#         sample_rate=16000,
#         hop_length=256,
#         win_length=1024,
#         n_mels=80,
#         n_fft=1024,
#         f_min=0.0,
#         f_max=8000.0,
#         power=1,
#         normalized=False,
#         min_max_energy_norm=True,
#         norm="slaney",
#         mel_scale="slaney",
#         compression=True
#     )


#     waveform = hifi_gan.decode_batch(spectrogram)
#     waveform = waveform.squeeze()
#     waveform = waveform[1330:] ## 0.06s
#     #torchaudio.save(f"LJSpeech_hifigan16K/{file_name}_vocoded.wav", waveform.unsqueeze(0), 16000)
#     print( len(signal) - len(waveform))
#     spectrogram_original = torch.stft(signal, n_fft=1024, hop_length=256, win_length=1024, return_complex=True)
#     spectrogram_vocoded = torch.stft(waveform, n_fft=1024, hop_length=256, win_length=1024, return_complex=True)

#     #print('difference between lengths:', len(signal) - len(waveform))
#     ## hifi-gan modifies temporal dimension by a bit, crop a few time stamps
#     min_T = min(spectrogram_original.shape[1], spectrogram_vocoded.shape[1])
#     spectrogram_original = spectrogram_original[:, :min_T]
#     spectrogram_vocoded = spectrogram_vocoded[:, :min_T]
#     freqs = torch.linspace(0, 16000 / 2, spectrogram_original.shape[0])
#     band_width = 1000
#     f_max = 8000
#     for start in range(0,f_max, band_width):
#         end = start+ band_width
#         mask = (freqs >=start) & (freqs< end)
#         #one vocoded band
#         spectrogram_combined = spectrogram_original.clone()
#         spectrogram_combined[mask, :] = spectrogram_vocoded[mask, :]
#         # 7 vocoded bands
#         # spectrogram_combined = spectrogram_vocoded.clone()
#         # spectrogram_combined[mask, :] = spectrogram_original[mask, :]
#         with torch.no_grad():
#             # MSE in-band (should match real)
#             mse_inband = torch.mean(
#                 (spectrogram_combined[mask] - spectrogram_original[mask]).abs()**2
#             ).item()

#             # MSE out-of-band (should match vocoded)
#             mse_outband = torch.mean(
#                 (spectrogram_combined[~mask] - spectrogram_vocoded[~mask]).abs()**2
#             ).item()

#             print(f"{file_name}  band {start}-{end}Hz  "
#                     f"IN={mse_inband:.6f}  OUT={mse_outband:.6f}")
#         combined_waveform = torch.istft(spectrogram_combined,n_fft=1024, hop_length=256, win_length=1024)
#         torchaudio.save(f"LJSpeech_vocoded16K/{file_name}_vocoded_{start}-{end}.wav", combined_waveform.unsqueeze(0), 16000)

#     #torchaudio.save('waveform_reconstructed.wav', waveforms.squeeze(1), 16000)

import torchaudio
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import librosa
import numpy as np
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

print("Loading HiFi-GAN...")
hifi_gan = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-libritts-16kHz",
    savedir="pretrained_models/hifigan_16k",
    run_opts={"device": str(device)},
)



wav_dir = "/mnt/QNAP/comdav/DATA/DATA/LJSpeech/wavs/"
output_dir = "LJSpeech_vocoded16K"
os.makedirs(output_dir, exist_ok=True)

file_names = []
metadata_path = "metadata/ljspeech_manipulated_metadata.txt"
if os.path.exists(metadata_path):
    with open(metadata_path, "r") as f:
        for line in f:
            file_names.append(line.strip().split(",")[0])
else:
    file_names = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]

file_names = file_names[:5000]
print(f"Total files to process: {len(file_names)}")

for file_name in tqdm(file_names, ascii=True, desc="Processing files"):
    full_path = os.path.join(wav_dir, file_name)
    if not os.path.exists(full_path):
        continue

    signal_np, _ = librosa.load(full_path, sr=16000)
    signal = torch.from_numpy(signal_np).float().to(device)

    spectrogram, _ = mel_spectogram(
        audio=signal,
        sample_rate=16000,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        n_fft=1024,
        f_min=0.0,
        f_max=8000.0,
        power=1,
        normalized=False,
        min_max_energy_norm=True,
        norm="slaney",
        mel_scale="slaney",
        compression=True,
    )

    waveform_voc = hifi_gan.decode_batch(spectrogram)
    waveform_voc = waveform_voc[1330:] 

    stft_real = torch.stft(
        signal,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        return_complex=True,
    )
    stft_voc = torch.stft(
        waveform_voc,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        return_complex=True,
    )

    freqs = torch.linspace(0, 8000, stft_real.shape[0]).to(device)

    for start in range(0, 8000, 1000):
        end = start + 1000
        mask = (freqs >= start) & (freqs < end)

        stft_combined = stft_real.clone()
        stft_combined[mask, :] = stft_voc[mask, :]


        wav_combined = torch.istft(
            stft_combined, n_fft=1024, hop_length=256, win_length=1024
        )

        out_name = f"{file_name}_vocoded_{start}-{end}.wav"
        torchaudio.save(
            os.path.join(output_dir, out_name), wav_combined.unsqueeze(0).cpu(), 16000
        )
