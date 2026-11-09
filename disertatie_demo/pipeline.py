from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T

HERE = Path(__file__).resolve().parent
_default_root = HERE if (HERE / "config.py").exists() else HERE.parent
REPO_ROOT = Path(os.environ.get("ADDVISOR_REPO_ROOT", _default_root)).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from audio import AudioProcessor
from models import TorchLogReg, load_mask_predictor
from viz import plot_mask, plot_spectrogram

KNN_CKPT = Path(os.environ.get("KNN_CKPT", HERE / "checkpoints" / "knn_model_attribution.joblib"))
DEFAULT_THRESHOLD = float(os.environ.get("DETECT_THRESHOLD", "0.5"))


@dataclass
class Result:
    is_fake: bool
    p_fake: float
    threshold: float
    sampling_rate: int
    audio: np.ndarray
    attributed_system: str | None = None
    attribution_confidence: float | None = None
    attribution_topk: list[tuple[str, float]] = field(default_factory=list)
    rel_audio: np.ndarray | None = None
    irr_audio: np.ndarray | None = None
    p_rel: float | None = None
    p_irr: float | None = None
    spec_fig: object | None = None
    mask_fig: object | None = None
    inv_mask_fig: object | None = None


class DeepfakePipeline:
    def __init__(self, setup_name: str = "realistic", device: str | torch.device | None = None):
        self.setup = config.get_setup(setup_name)
        self.device = config.get_device(device) if not isinstance(device, torch.device) else device

        self.ap = AudioProcessor(device=self.device)
        self.logreg = TorchLogReg(self.setup.detector_ckpt).to(self.device).eval()
        self.unet = load_mask_predictor(self.setup.mask_ckpt, self.device)

        if not KNN_CKPT.exists():
            raise FileNotFoundError(
                f"k-NN attribution checkpoint not found at {KNN_CKPT}. "
                "Run run_model_attribution_new.py first."
            )
        bundle = joblib.load(KNN_CKPT)
        self.knn = bundle["knn"]
        self.knn_scaler = bundle["scaler"]
        self.knn_labels: dict[int, str] = bundle["labels"]
        self.bert_layer: int = bundle["layer"]
        self.bert_model_id: str = bundle["feature_model"]
        self.n_systems: int = bundle.get("n_systems", len(self.knn_labels))

        self._bert = None
        self._bert_fe = None

    def _ensure_bert(self):
        if self._bert is None:
            from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

            self._bert_fe = AutoFeatureExtractor.from_pretrained(self.bert_model_id)
            self._bert = Wav2Vec2BertModel.from_pretrained(self.bert_model_id).to(self.device).eval()
            for p in self._bert.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def _bert_embedding(self, wave_16k: torch.Tensor) -> np.ndarray:
        self._ensure_bert()
        audio = wave_16k.detach().cpu().numpy().astype(np.float32)
        inputs = self._bert_fe(audio, sampling_rate=self.ap.sampling_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self._bert(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[self.bert_layer]
        return hidden.mean(dim=1).squeeze(0).float().cpu().numpy()

    def _load_full_16k(self, path: str | Path) -> torch.Tensor:
        audio, sr = torchaudio.load(str(path))
        if audio.ndim > 1:
            audio = audio.mean(dim=0)
        if sr != self.ap.sampling_rate:
            audio = T.Resample(sr, self.ap.sampling_rate)(audio)
        return audio

    @torch.no_grad()
    def _detector_prob(self, wave: torch.Tensor) -> float:
        if wave.ndim == 1:
            wave = wave.unsqueeze(0)
        pooled = self.ap.extract_features(wave.to(self.device)).mean(dim=1)
        return self.logreg(pooled)[1].item()

    @torch.no_grad()
    def attribute(self, wave_16k: torch.Tensor, topk: int = 5):
        emb = self._bert_embedding(wave_16k).reshape(1, -1)
        emb = self.knn_scaler.transform(emb)
        proba = self.knn.predict_proba(emb)[0]
        classes = self.knn.classes_
        order = np.argsort(proba)[::-1]
        ranked = [(self.knn_labels[int(classes[i])], float(proba[i])) for i in order if proba[i] > 0]
        if not ranked:
            ranked = [(self.knn_labels[int(classes[order[0]])], float(proba[order[0]]))]
        return ranked[:topk]

    @torch.no_grad()
    def explain(self, wave_5s: torch.Tensor):
        wave = wave_5s.to(self.device)
        _, mag, phase = self.ap.compute_stft(wave.unsqueeze(0), has_batch_dim=True)

        mask = self.unet(mag.unsqueeze(1)).squeeze(1)
        Tmax = mask.shape[1]
        mag, phase = mag[:, :Tmax, :], phase[:, :Tmax, :]
        spec_phase = torch.exp(1j * phase)

        rel_wave = self.ap.compute_invert_stft((mask * mag) * spec_phase)
        irr_wave = self.ap.compute_invert_stft(((1 - mask) * mag) * spec_phase)

        return {
            "mag": mag[0].cpu(),
            "mask": mask[0].cpu(),
            "rel_audio": rel_wave.squeeze(0).cpu().numpy(),
            "irr_audio": irr_wave.squeeze(0).cpu().numpy(),
            "p_rel": self._detector_prob(rel_wave),
            "p_irr": self._detector_prob(irr_wave),
        }

    def run(self, audio_path: str | Path, threshold: float = DEFAULT_THRESHOLD) -> Result:
        wave_5s, _ = self.ap.load_audio(audio_path)
        p_fake = self._detector_prob(wave_5s)
        is_fake = p_fake >= threshold

        result = Result(
            is_fake=is_fake,
            p_fake=p_fake,
            threshold=threshold,
            sampling_rate=self.ap.sampling_rate,
            audio=wave_5s.cpu().numpy(),
        )
        if not is_fake:
            return result

        wave_full = self._load_full_16k(audio_path)
        result.attribution_topk = self.attribute(wave_full)
        result.attributed_system, result.attribution_confidence = result.attribution_topk[0]

        ex = self.explain(wave_5s)
        result.rel_audio = ex["rel_audio"]
        result.irr_audio = ex["irr_audio"]
        result.p_rel = ex["p_rel"]
        result.p_irr = ex["p_irr"]
        result.spec_fig = plot_spectrogram(ex["mag"], title="log-magnitude spectrogram", log=True)
        result.mask_fig = plot_mask(ex["mask"], title="relevance mask")
        result.inv_mask_fig = plot_mask(1 - ex["mask"], title="1 - mask")
        return result
