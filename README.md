# ADDvisor — explaining audio deepfake detection

ADDvisor is an explainability pipeline for audio deepfake detection. It pairs a
frozen **detector** (a `wav2vec2-xls-r-2b` backbone + logistic-regression head
that scores audio as real vs. fake) with a U-Net **mask predictor** that
highlights the time–frequency regions driving the "fake" decision. The mask is a
listenable, LMAC-style explanation: keep the *relevant* part of the spectrogram
and the detector's decision is preserved; keep the *irrelevant* part and it
flips.

The pipeline is studied under three **setups**:

| setup           | data                              | manipulation                          |
|-----------------|-----------------------------------|---------------------------------------|
| `realistic`     | `mlaad_subset/`                   | real-world TTS deepfakes (MLAAD)      |
| `freq-swapped`  | `LJSpeech_freq_swap/`             | one 1 kHz band replaced by a vocoder  |
| `time-swapped`  | `LJSpeech_time_swap/`             | the 3–5 s segment replaced by a vocoder |

## Layout

**Library modules** (import these):

| module          | what's in it                                                        |
|-----------------|---------------------------------------------------------------------|
| `config.py`     | all paths, STFT params, and the three `Setup`s (single source of truth) |
| `audio.py`      | `AudioProcessor` — load, STFT/ISTFT, wav2vec2 features              |
| `models.py`     | `UNet` (mask predictor), `TorchLogReg`, `SpectrogramCNN`, backbone loader |
| `detector.py`   | `SpectralDetector` / `WaveformDetector` + `build_detector`          |
| `losses.py`     | `LMACLoss` (the mask-predictor training objective)                  |
| `data.py`       | `AudioDataset`, metadata readers, collate fns, corpus finders       |
| `explainers.py` | `build_explainer` — trained UNet + random + captum baselines        |
| `metrics.py`    | fidelity, faithfulness, AD/AI/AG, localization (+ evaluators)       |
| `viz.py`        | `plot_mask` / `plot_spectrogram` / `plot_features`, GT masks        |
| `vocoder.py`    | HiFi-GAN vocoding + freq/time swap generators                       |

**Command-line scripts** (`python <script>.py --help`):

| script               | purpose                                                       |
|----------------------|---------------------------------------------------------------|
| `evaluate.py`        | evaluate an explainer on a setup (fidelity/faithfulness/…)    |
| `train_mask.py`      | train the U-Net mask predictor                                |
| `train_detector.py`  | train the detector: `logreg`, `timeswap` or `cnn`             |
| `generate_data.py`   | regenerate frequency / time swap datasets                     |
| `export_masks.py`    | dump masks/spectrograms (`npy`) or masked features + metrics  |
| `tsne.py`            | t-SNE of real/fake/relevant/irrelevant features               |
| `streamlit_app.py`   | interactive demo (`streamlit run streamlit_app.py`)           |

**Paper figures & analysis:**

| script                            | purpose                                                  |
|-----------------------------------|----------------------------------------------------------|
| `show_explanations_interspeech26.py` | Streamlit grid of input + explanations per setup       |
| `generate_tsne_2.py`              | Streamlit 2-panel t-SNE figure (realistic + freq)        |
| `train_logReg_selectKBest.py`     | SelectKBest top-K-vs-rest EER feature-importance analysis |

**Data & checkpoints** (git-ignored, large): `ckpts/` (trained UNets +
logistic-regression `.joblib`s), `models/wav2vec2-xls-r-2b_truncated/`,
`metadata/`, and the wav directories above. Outputs go to `runs/` (training),
`output/` (cached explanations), `npy/`, `features/`, `visualizations/`.

## Install

```bash
pip install -r requirements.txt        
```

The truncated wav2vec2 backbone is expected at
`models/wav2vec2-xls-r-2b_truncated/`; if absent, the public
`facebook/wav2vec2-xls-r-2b` is used. Trained checkpoints live under `ckpts/`
(see `config.SETUPS`).

## Quickstart

```bash
# Evaluate the trained mask predictor on the frequency-swap setup
python evaluate.py --setup freq-swapped --explanation optimized

# Compare against a gradient baseline
python evaluate.py --setup time-swapped --explanation saliency

# Train a mask predictor from scratch
python train_mask.py --setup freq-swapped --epochs 100 --batch-size 15

# Dump masks and report LMAC metrics
python export_masks.py npy      --setup realistic
python export_masks.py features --setup realistic
python tsne.py --setup realistic

# Interactive demo
streamlit run streamlit_app.py
```

## Programmatic use



```python
import config
from audio import AudioProcessor
from detector import build_detector
from explainers import build_explainer

setup = config.get_setup("freq-swapped")
ap = AudioProcessor()
detector = build_detector(setup, ap)
explain = build_explainer(setup, "optimized", detector)

_, magnitude, phase = ap.compute_stft(ap.load_audio("some.wav")[0].unsqueeze(0))
mask = explain(magnitude, phase)          
```



```python
from evaluate import evaluate
print(evaluate("freq-swapped", "optimized"))
# {'Fidelity': ..., 'Faithfulness': ..., 'Average Drop': ..., 'Localization': ...}
```

## Configuration

Change paths, STFT parameters or checkpoint selection in **one place** —
`config.py`. Each `Setup` bundles its data directory, evaluation file list, mask
checkpoint and detector checkpoint.
