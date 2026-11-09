from __future__ import annotations

import os
import tempfile
import types
from pathlib import Path

os.environ.setdefault("STREAMLIT_SERVER_ENABLE_FILE_WATCHER", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import streamlit as st
import torch

from pipeline import DeepfakePipeline

if isinstance(torch.classes, types.ModuleType):
    torch.classes.__path__ = []

st.set_page_config(layout="wide", page_title="xAI analysis")

HERE = Path(__file__).resolve().parent


def available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.insert(0, "cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        devices.insert(0, "mps")
    return devices


@st.cache_resource(show_spinner="Loading models (detector, w2v-bert, k-NN, U-Net) ...")
def load_pipeline(device: str) -> DeepfakePipeline:
    return DeepfakePipeline(setup_name="realistic", device=device)


st.sidebar.title("ADDvisor")
st.sidebar.caption("detection, attribution, explainability")

devices = available_devices()
device = st.sidebar.radio("Compute device", options=devices, index=0)
threshold = st.sidebar.slider("Detection threshold: P(fake) above this is fake", 0.05, 0.95, 0.5, 0.05)

example_dirs = {
    "fake": HERE / "examples" / "fake",
    "real": HERE / "examples" / "real",
}
examples = {}
for label, d in example_dirs.items():
    if d.exists():
        for p in sorted(d.glob("*.wav")):
            examples[f"{label} - {p.name}"] = p
example_choice = st.sidebar.selectbox("or try an example clip", ["(upload my own)"] + list(examples), index=0)


st.title("xAI analysis")
st.markdown(
    "Upload a speech clip. If it is a deepfake, the app names the most likely "
    "generator model (k-NN over w2v-bert-2.0 features) and shows where the "
    "detector found evidence (U-Net spectrogram mask)."
)

uploaded = st.file_uploader("Upload audio (wav / flac / mp3)", type=["wav", "flac", "mp3", "ogg", "m4a"])

audio_path = None
tmp_to_cleanup = None
if uploaded is not None:
    suffix = Path(uploaded.name).suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.flush()
    tmp.close()
    audio_path = tmp.name
    tmp_to_cleanup = tmp.name
elif example_choice != "(upload my own)":
    audio_path = str(examples[example_choice])

run = st.button("Analyze", type="primary", disabled=audio_path is None)

if run and audio_path is not None:
    pipe = load_pipeline(device)
    st.audio(audio_path, format="audio/wav")

    with st.spinner("Running detector, attribution, explainability ..."):
        res = pipe.run(audio_path, threshold=threshold)

    if not res.is_fake:
        st.success(f"Real (bonafide) - detector P(fake) = {res.p_fake:.3f} (threshold {res.threshold:.2f})")
        st.info("The clip is judged authentic, so attribution and explainability are skipped.")
    else:
        st.error(f"Fake (synthetic / manipulated) - detector P(fake) = {res.p_fake:.3f} (threshold {res.threshold:.2f})")

        st.subheader("Model attribution")
        top = res.attribution_topk
        st.markdown(
            f"Most likely generator: `{res.attributed_system}` "
            f"(k-NN confidence {res.attribution_confidence:.2f})"
        )
        st.bar_chart({name: conf for name, conf in top})
        with st.expander("Top-k candidates"):
            for name, conf in top:
                st.write(f"- `{name}` - {conf:.3f}")

        st.subheader("Explainability (U-Net relevance mask)")
        c1, c2, c3 = st.columns(3)
        c1.pyplot(res.spec_fig)
        c2.pyplot(res.mask_fig)
        c3.pyplot(res.inv_mask_fig)

        a, b = st.columns(2)
        a.markdown("Relevant reconstruction (mask x audio)")
        a.audio(res.rel_audio, sample_rate=res.sampling_rate)
        b.markdown("Irrelevant reconstruction ((1 - mask) x audio)")
        b.audio(res.irr_audio, sample_rate=res.sampling_rate)

        st.markdown(
            "Faithfulness check (re-scoring the masked reconstructions): "
            f"original P(fake) = {res.p_fake:.3f}, "
            f"relevant P(fake) = {res.p_rel:.3f}, "
            f"irrelevant P(fake) = {res.p_irr:.3f}. "
            "A good explanation keeps the relevant part fake-like while the "
            "irrelevant part drops toward real."
        )

    if tmp_to_cleanup:
        try:
            os.unlink(tmp_to_cleanup)
        except OSError:
            pass
