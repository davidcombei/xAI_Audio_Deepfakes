from __future__ import annotations

import types

import streamlit as st
import torch

import config
from audio import AudioProcessor
from data import read_paths
from models import TorchLogReg, load_mask_predictor
from viz import plot_mask, plot_spectrogram

if isinstance(torch.classes, types.ModuleType):
    torch.classes.__path__ = []

st.set_page_config(layout="wide", page_title="ADDvisor")


@st.cache_resource(show_spinner="Loading models ...")
def load_pipeline(setup_name: str):
    setup = config.get_setup(setup_name)
    device = config.get_device()
    ap = AudioProcessor(device=device)
    model = load_mask_predictor(setup.mask_ckpt, device)
    logreg = TorchLogReg(setup.detector_ckpt).to(device)
    paths = read_paths(setup.filelist)
    return setup, ap, model, logreg, paths


@torch.no_grad()
def explain(ap, model, logreg, path):
    wave, _ = ap.load_audio(config.PROJECT_ROOT / path)
    wave = wave.to(ap.device)
    _, mag, phase = ap.compute_stft(wave.unsqueeze(0), has_batch_dim=True)

    mask = model(mag.unsqueeze(1)).squeeze(1)              # (1, F, T)
    Tmax = mask.shape[1]
    mag, phase = mag[:, :Tmax, :], phase[:, :Tmax, :]
    spec_phase = torch.exp(1j * phase)

    rel_wave = ap.compute_invert_stft((mask * mag) * spec_phase)
    irr_wave = ap.compute_invert_stft(((1 - mask) * mag) * spec_phase)

    def prob(w):
        if w.ndim == 1:
            w = w.unsqueeze(0)
        pooled = ap.extract_features(w).mean(dim=1)  # (1, D)
        return logreg(pooled)[1].item()

    return {
        "audio": wave.cpu().numpy(),
        "rel_audio": rel_wave.squeeze(0).cpu().numpy(),
        "mag": mag[0].cpu(),
        "mask": mask[0].cpu(),
        "p_orig": prob(wave.unsqueeze(0)),
        "p_rel": prob(rel_wave),
        "p_irr": prob(irr_wave),
    }


st.title("ADDvisor - explaining audio deepfake detection")

setup_name = st.sidebar.selectbox("Setup", config.SETUP_NAMES)
setup, ap, model, logreg, paths = load_pipeline(setup_name)
n = st.sidebar.slider("Clips to show", 1, min(50, len(paths)), 5)
st.sidebar.caption(f"{len(paths)} clips available in `{setup.filelist.name}`")
st.sidebar.caption("Prediction = detector probability of class 1.")

for path in paths[:n]:
    st.subheader(path)
    out = explain(ap, model, logreg, path)

    a, b = st.columns(2)
    a.markdown("**Original**")
    a.audio(out["audio"], sample_rate=config.SAMPLING_RATE)
    b.markdown("**Relevant (mask) reconstruction**")
    b.audio(out["rel_audio"], sample_rate=config.SAMPLING_RATE)

    c1, c2, c3 = st.columns(3)
    c1.pyplot(plot_spectrogram(out["mag"], title="spectrogram", log=True))
    c2.pyplot(plot_mask(out["mask"], title="mask"))
    c3.pyplot(plot_mask(1 - out["mask"], title="1 - mask"))

    st.write(
        f"original = {out['p_orig']:.3f}  |  "
        f"relevant = {out['p_rel']:.3f}  |  "
        f"irrelevant = {out['p_irr']:.3f}"
    )
    st.markdown("---")
