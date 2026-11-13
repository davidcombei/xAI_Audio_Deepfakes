from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

import config


SWAP_N_FFT = config.SWAP_N_FFT
SWAP_HOP = config.SWAP_HOP
SWAP_WIN = config.SWAP_WIN
HIFIGAN_LATENCY = 1330  


def load_hifigan(device=None, savedir=None, source="speechbrain/tts-hifigan-libritts-16kHz"):
    """Load the speechbrain HiFi-GAN vocoder (downloaded on first use)."""
    from speechbrain.inference.vocoders import HIFIGAN

    device = config.get_device(device)
    savedir = str(savedir or (config.PROJECT_ROOT / "pretrained_models" / "hifigan_16k"))
    return HIFIGAN.from_hparams(source=source, savedir=savedir, run_opts={"device": str(device)})


def vocode(signal: torch.Tensor, hifi_gan, sr: int = config.SAMPLING_RATE) -> torch.Tensor:
    """Re-synthesise a mono waveform through HiFi-GAN (returns a 1-D tensor)."""
    from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

    spectrogram, _ = mel_spectogram(
        audio=signal, sample_rate=sr, hop_length=256, win_length=1024, n_mels=80,
        n_fft=1024, f_min=0.0, f_max=8000.0, power=1, normalized=False,
        min_max_energy_norm=True, norm="slaney", mel_scale="slaney", compression=True,
    )
    waveform = hifi_gan.decode_batch(spectrogram).squeeze()
    return waveform[HIFIGAN_LATENCY:]


def _swap_stft(x):
    return torch.stft(x, n_fft=SWAP_N_FFT, hop_length=SWAP_HOP, win_length=SWAP_WIN, return_complex=True)


def _swap_istft(x):
    return torch.istft(x, n_fft=SWAP_N_FFT, hop_length=SWAP_HOP, win_length=SWAP_WIN)


def generate_freq_swaps(
    wav_dir, out_dir, file_list, *, band_width: int = 1000, f_max: int = 8000,
    sr: int = config.SAMPLING_RATE, device=None,
):
    """Write one freq-band-swapped clip per (file, band) into ``out_dir``."""
    import librosa

    device = config.get_device(device)
    hifi_gan = load_hifigan(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for file_name in tqdm(file_list, ascii=True, desc="freq swaps"):
        full_path = Path(wav_dir) / file_name
        if not full_path.exists():
            continue
        signal = torch.from_numpy(librosa.load(str(full_path), sr=sr)[0]).float().to(device)
        voc = vocode(signal, hifi_gan, sr)

        stft_real, stft_voc = _swap_stft(signal), _swap_stft(voc)
        freqs = torch.linspace(0, f_max, stft_real.shape[0]).to(device)
        for start in range(0, f_max, band_width):
            band = (freqs >= start) & (freqs < start + band_width)
            combined = stft_real.clone()
            combined[band, :] = stft_voc[band, :]
            wav = _swap_istft(combined)
            out = out_dir / f"{file_name}_vocoded_{start}-{start + band_width}.wav"
            torchaudio.save(str(out), wav.unsqueeze(0).cpu(), sr)


