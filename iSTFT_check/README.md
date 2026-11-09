# iSTFT_check — explanations with and without the iSTFT

The main ADDvisor pipeline explains the detector on the **spectrogram**:

    spectrogram -> mask -> iSTFT (back to audio) -> wav2vec2 -> logistic regression -> fake / real

This folder **skips the STFT and the iSTFT**. It explains the **waveform**
directly and asks two questions:

1. Does the iSTFT itself cause problems?
2. On the time-swapped clips (3–5 s replaced by a HiFi-GAN vocoder), do the
   explanations find the manipulated part?

Nothing in the main pipeline is changed; these scripts only import it.

## Files (read them in this order)

| file | what it does |
|------|--------------|
| `common.py` | shared helpers: constants, loading the detector (spectrogram input / waveform input), frames ↔ samples, ground truth, AUC |
| `waveform_explainers.py` | the waveform explanations: `optimized`, `trained`, `random`, `saliency`, `input-x-gradient`, `gradient-shap` |
| `roundtrip_check.py` | does STFT → iSTFT change the audio or the detector's answer? (no explanations) |
| `compare.py` | **the evaluation**: TF vs waveform explanations, scored with AUC, fidelity and faithfulness |
| `waveform_model.py` | `FrameMaskNet` (a trainable waveform mask predictor) and its loss |
| `train_waveform_mask.py` | trains `FrameMaskNet` (the waveform version of `train_mask.py`) |
| `position_control.py` | moves the swap to other positions: does the trained mask follow the swap, or just its usual position? |

Words used everywhere: **TF** = time-frequency (spectrogram, through the iSTFT),
**wf** = waveform (no iSTFT), **frame** = one spectrogram column (~20 ms, 248 per
clip), **time profile** = one number per frame.

## The three scores

| score | meaning | for |
|-------|---------|-----|
| **AUC** (`auc_tf`, `auc_wf`) | does the explanation point at the manipulated 3–5 s? 1 = perfectly, 0.5 = chance, < 0.5 = it points at the real part. The TF explanation is averaged over frequency first. | every method, time-swapped setup only |
| **fidelity** (`fid_tf`, `fid_wf`) | keep only what the mask keeps: does the detector still give the same answer? (1 = yes) | masks: `optimized`, `trained`, `random` |
| **faithfulness** (`faith_tf`, `faith_wf`) | remove what the mask keeps: how much does P(fake) drop? (higher = the mask matters more) | masks |

`compare.py` writes two summaries: one over all files, and one over the
**well-posed** files only. A file is well-posed when the detector calls it fake
and the original clip is at least 5 s long. The clips were padded with silence
to 5 s *before* the swap, so in 357 of the 5000 clips (original < 3 s) nothing
was actually manipulated, even though the ground truth marks 3–5 s as fake.

## Run

From the repo root. The system `python3` starts much faster than the NAS `.venv`.

```bash
# 1. does the iSTFT change anything? (100-file list; --all for all 5000 clips)
python3 iSTFT_check/roundtrip_check.py

# 2. the evaluation on the 100-file list
python3 iSTFT_check/compare.py

#    ... or on all 5000 clips: one part per GPU, then merge the parts
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python3 iSTFT_check/compare.py --all --shard $i --num-shards 8 &
done; wait
python3 iSTFT_check/compare.py --all --summarize-only

# 3. train the waveform mask predictor, then evaluate it
#    (--gpu_ids is needed: this machine's default accelerate config pins its own GPUs)
accelerate launch --multi_gpu --gpu_ids 0,1,2,3,4,5,6,7 --num_processes 8 \
    iSTFT_check/train_waveform_mask.py --setup time-swapped --epochs 400 --batch-size 4 --resume
python3 iSTFT_check/compare.py --methods trained \
    --waveform-mask-ckpt runs/waveform_mask/time-swapped/FrameMaskNet_epoch_175_loss_1.1146.pth

# 4. does the trained mask follow the swap, or its usual position?
python3 iSTFT_check/position_control.py \
    --waveform-mask-ckpt runs/waveform_mask/time-swapped/FrameMaskNet_epoch_175_loss_1.1146.pth
```

Results go to `output/istft_check/`:
- `roundtrip_<setup>_<subset|all>.csv`
- `<setup>_<subset|all>/`: `per_file*.csv`, `summary*.csv`, `summary_wellposed*.csv` and `plots/`
- `position_control_<checkpoint>.csv`

## How "without the iSTFT" is made comparable

- **Same detector.** The same logistic regression on the same wav2vec2
  features. It is fed the waveform instead of `iSTFT(spectrogram)`.
- **Same frames.** The pipeline's STFT uses 644-sample frames, one every 322
  samples, so every audio sample lies inside exactly two frames.
  `frames_to_samples` gives each sample the average of its two frames' mask
  values. That is exactly what scaling whole spectrogram frames and running the
  iSTFT does, so a waveform mask is the same operation as a frequency-flat
  spectrogram mask, minus the iSTFT.
- **Same kind of explanation.**
  - `optimized` fits a mask per clip with the same LMAC loss as the U-Net
    (100 Adam steps, lr 0.1, size weight 1).
  - `trained` (`FrameMaskNet`) is trained like the U-Net (`train_mask.py`): same
    data, loss and learning rates. A learned filterbank on the STFT's frame grid
    replaces the STFT.
  - Both are compared with the trained U-Net on the TF side.
- **Faster wav2vec2.** The detector reads layer 9 only, so layers above it are
  removed. The features are identical; everything runs ~4× faster.

## Results (time-swapped, all 5000 clips, 2026-10-07)

### 1. Does the iSTFT cause problems? — `roundtrip_check.py`

| case (TF path vs waveform path) | SNR (dB) | P(fake) TF | P(fake) wf | same decision |
|---------------------------------|---------:|-----------:|-----------:|--------------:|
| identity (pipeline round trip)  | 35.9 | 0.971 | 0.968 | 98.1 % |
| keep 3–5 s                      | 28.5 | 0.935 | 0.934 | 99.9 % |
| keep 0–3 s                      | 45.5 | 0.128 | 0.111 | 96.5 % |
| round trip, no crop             | 120 (exact) | 0.968 | 0.968 | 100 % |
| round trip, Nyquist bin dropped | 44.8 | 0.971 | 0.968 | 98.1 % |
| round trip, last frame dropped  | 38.0 | 0.968 | 0.968 | 100 % |

- **The iSTFT itself is exact.** Without the crop, STFT → iSTFT gives back the
  same audio and the same detector output.
- **The pipeline's crop is not harmless.** `compute_stft` drops the top
  frequency bin (8 kHz, "Nyquist") and the last frame, and the iSTFT puts them
  back as zeros. Dropping the Nyquist bin alone flips 1.9 % of the decisions,
  all on clips shorter than 3 s, whose score sits right at the boundary. It
  also pushes most scores toward "fake" (the earlier, more detailed version of
  this check measured a median of +1.6 logit). The last frame only affects the
  final 30 ms.
- **Suggested fix (not applied):** in `compute_invert_stft`, put back the
  original Nyquist bin and last frame instead of zeros.

### 2. Do the explanations find the manipulated 3–5 s? — `compare.py`

| method | AUC TF | AUC wf | fidelity TF / wf | faithfulness TF / wf |
|--------|-------:|-------:|-----------------:|---------------------:|
| saliency | 0.65 | 0.66 | — | — |
| input-x-gradient | 0.55 | 0.55 | — | — |
| gradient-shap | 0.55 | 0.55 | — | — |
| optimized (TF: U-Net, wf: per-clip mask) | 0.15 | 0.54 | 0.98 / 1.00 | 0.94 / 0.92 |
| trained (TF: U-Net, wf: FrameMaskNet epoch 175) | 0.15 | 0.82 | 0.98 / 0.98 | 0.94 / 0.93 |
| random | 0.50 | 0.51 | 0.98 / 0.98 | −0.03 / −0.02 |

Well-posed files only (3754 / 5000):

| method | AUC TF | AUC wf | fidelity TF / wf | faithfulness TF / wf |
|--------|-------:|-------:|-----------------:|---------------------:|
| saliency | 0.61 | 0.61 | — | — |
| input-x-gradient | 0.54 | 0.54 | — | — |
| optimized | 0.17 | 0.61 | 1.00 / 1.00 | 1.00 / 0.98 |
| trained | 0.17 | 0.88 | 1.00 / 1.00 | 1.00 / 0.98 |

- **Gradient explanations behave the same with and without the iSTFT.** Their
  AUCs match to ±0.01. The earlier, more detailed version of this check found
  their time profiles correlate at r ≈ 0.98. They only weakly find the
  manipulated part; saliency is best at AUC ≈ 0.61–0.66.
- **The U-Net does not explain *when*.** It keeps the < 1 kHz band over the
  whole clip, and its time profile points away from 3–5 s in every file
  (AUC 0.15–0.17). This comes from the trained U-Net, not from the iSTFT.
- **Fidelity and faithfulness are ≈ 1 for very different masks** (U-Net,
  per-clip mask, FrameMaskNet). The detector is extremely confident
  (|logit| ≈ 50), so they don't tell a good localization from a bad one.

### 3. The trained waveform mask: real detection or position? — `position_control.py`

`FrameMaskNet` epoch 175 gets AUC 0.88, but all training swaps sit at 3–5 s.
Moving the swap (75 clips ≥ 5 s, the detector calls all of them fake):

| swap at | trained AUC | U-Net AUC | waveform saliency AUC |
|---------|------------:|----------:|----------------------:|
| 3–5 s (training position) | **0.88** | 0.17 | 0.60 |
| 2–4 s | 0.73 | 0.23 | 0.60 |
| 1–3 s | 0.65 | 0.24 | 0.58 |
| 0–2 s | **0.30** | 0.27 | 0.49 |

The trained mask learned the **position**: it removes the opening of every clip
wherever the swap is. Saliency has no position prior and keeps about the same
score. So neither trained explainer (TF or waveform) can be credited with time
localization on this data. A fair test needs training swaps at random positions
and lengths, for both the U-Net and FrameMaskNet; that has not been done.
`FrameMaskNet` was trained to epoch 179 (`runs/waveform_mask/time-swapped/`,
continue with `--resume`).
