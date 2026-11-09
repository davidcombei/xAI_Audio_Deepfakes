# new_masking — fill the masked-out part with the real clip, not with zeros

The ADDvisor U-Net is trained with the LMAC loss: keeping the masked part of the
spectrogram should keep the detector's decision, and removing it should flip it.
In the original pipeline "removing" means **multiplying by zero**. Here the
masked-out bins are **filled with the same bins of the bona fide clip the fake
was made from**:

| | keep (LMAC "relevant") | remove (LMAC "irrelevant") |
|---|---|---|
| zero fill (original) | `m·X_fake` | `(1−m)·X_fake` |
| real fill (this experiment) | `m·X_fake + (1−m)·X_real` | `(1−m)·X_fake + m·X_real` |

`X` is the complex STFT (magnitude and phase). With a 0/1 mask, "keep" is the
fake inside the mask and the real clip outside it, and "remove" is the other way
round. Outside the manipulated region `X_fake = X_real`, so the mask value there
changes neither input.

The U-Net was retrained with this loss on both swap setups and evaluated on
**all** files (40 000 freq-swapped, 5 000 time-swapped clips) with fidelity,
faithfulness and AUC, under both fills.

## Why

Zero masking is itself out of distribution for the detector. On 60 clips per
setup (`sanity_check.py`, `results/logs/sanity_check.txt`), the detector calls
these inputs fake:

| input | freq detector | time detector |
|---|---:|---:|
| fake clip | 1.00 | 0.97 |
| bona fide clip | 0.02 | 0.02 |
| bona fide clip, one 1 kHz band set to zero | **0.98** | — |
| bona fide clip, everything but one 1 kHz band set to zero | **0.62** | — |
| bona fide clip, 3–5 s set to zero | — | 0.13 |
| bona fide clip, 0–3 s set to zero | — | **0.87** |

Zeroing parts of a *real* clip makes it look fake, so under zero fill the
fidelity/faithfulness scores mix "found the manipulation" with "made the input
look odd". With real fill the filled-in part is real speech.

## Setup

* **Bona fide reference.** Both swap datasets were built from
  `LJSpeech_vocoded22K/<id>.wav`. Checked: outside the swapped segment the
  time swaps equal the original up to 16-bit rounding (max |Δ| = 1.5e-5), and
  97–99 % of the fake−real difference energy lies inside the named band for the
  freq swaps. Re-saving, STFT→ISTFT round trips and dither leave the detector's
  verdict on bona fide clips unchanged (median logit −41 freq / −48 time detector), so the detector reacts to
  the vocoder, not to the processing.
* **Only the loss changes.** Same U-Net, same input (the fake's 512 × 248 STFT
  magnitude), same LMAC terms, learnable loss weights and initial values, Adam
  lr 3e-5 / 1e-4, batch 15 per GPU. The U-Net never sees the bona fide clip.
  With `fill="zero"` the new loss reproduces `losses.LMACLoss` (171.1176 vs
  171.1177 on a test batch).
* **Runs.** freq-swapped: 5 GPUs × 15 (effective batch 75), 40 epochs.
  time-swapped: 3 GPUs × 15 (45), 150 epochs (≈ the 173 × 5 000 clips behind the
  original epoch-173 checkpoint). Run time ≈ 10 h on 8 V100s, both in parallel.
* **All files.** The runs started with 500 utterance ids held out
  (`runs/held_out_ids_early_epochs.txt`). From freq epoch 24 / time epoch 118 on,
  they were resumed on all files, so every file was in training for the last
  17 / 33 epochs. Evaluation uses all files.
* **BatchNorm statistics.** The running BatchNorm averages saved with the freq
  epoch-40 checkpoint did not match the batch statistics used in training. On
  240 random clips, removing the mask flipped 52 % of decisions in eval mode,
  against 98 % with batch statistics (epoch 30's checkpoint: 99.6 %, fine).
  `recalibrate_bn.py` recomputes the averages over one pass of the data
  without changing any weights. Both new checkpoints are evaluated in this
  recalibrated form (`*_bn_recalibrated.pth`). The time checkpoint was already
  fine (94.6 % before and after). Before recalibration, freq real-fill
  faithfulness was 0.507 (`results/*/summary_before_bn_recalibration.txt`).
  The original U-Net is evaluated exactly as shipped.

## Results (all files)

Fidelity: share of clips where keeping only the masked part keeps the decision.
Faithfulness: mean of p_fake(original) − p_fake(masked part removed), as in
`evaluate.py`. AUC: per-clip ROC-AUC of the mask against the ground-truth region.

### freq-swapped (40 000 clips, all called fake)

| mask | Fidelity, zero fill | Faithfulness, zero fill | Fidelity, real fill | Faithfulness, real fill | AUC |
|---|---:|---:|---:|---:|---:|
| U-Net, real fill (**new**) | 1.000 | 0.020 | **1.000** | **0.990** | **0.741** |
| U-Net, zero fill (original) | 1.000 | **1.000** | 0.259 | 0.000 | 0.503 |
| ground truth | 0.757 | 0.001 | 0.994 | 0.144 | 1.000 |

Per swapped band (`results/freq-swapped/summary_per_band.csv`):

| band (kHz) | new: Fid / Faith, real fill | new AUC | original AUC | ground truth: Fid / Faith, real fill | ground truth: Fid / Faith, zero fill |
|---|---:|---:|---:|---:|---:|
| 0–1 | 1.000 / 0.989 | 0.750 | 0.750 | 1.000 / 0.000 | 1.000 / 0.000 |
| 1–2 | 1.000 / 0.997 | 0.747 | 0.553 | 0.996 / 0.057 | 1.000 / 0.000 |
| 2–3 | 1.000 / 0.996 | 0.721 | 0.423 | 0.995 / 0.172 | 1.000 / 0.000 |
| 3–4 | 1.000 / 0.993 | 0.778 | 0.508 | 0.998 / 0.249 | 0.987 / 0.000 |
| 4–5 | 1.000 / 0.983 | 0.749 | 0.495 | 0.996 / 0.080 | 0.759 / 0.000 |
| 5–6 | 1.000 / 0.979 | 0.697 | 0.424 | 0.999 / 0.079 | 0.751 / 0.000 |
| 6–7 | 1.000 / 0.990 | 0.720 | 0.415 | 0.995 / 0.035 | 0.280 / 0.000 |
| 7–8 | 1.000 / 0.992 | 0.765 | 0.460 | 0.973 / 0.481 | 0.276 / 0.010 |

### time-swapped (5 000 clips, 97.4 % called fake)

| mask | Fidelity, zero fill | Faithfulness, zero fill | Fidelity, real fill | Faithfulness, real fill | AUC |
|---|---:|---:|---:|---:|---:|
| U-Net, real fill (**new**) | 0.976 | 0.089 | **0.999** | **0.932** | **0.678** |
| U-Net, zero fill (original) | 0.979 | **0.938** | 0.997 | 0.070 | 0.372 |
| ground truth | 0.961 | 0.843 | 0.999 | 0.935 | 1.000 |

Real-fill faithfulness tops out at ≈ 0.93 here. In 357 clips the original is
shorter than 3 s, so 3–5 s is silence in both clips and nothing was swapped;
no mask can flip those.

### Figures (`results/figures/`)

* `mask_profiles_freq.png` — mask averaged over time, per swapped band. The
  original U-Net keeps < 2 kHz whatever band was swapped. The new one peaks on
  the swapped band in all 8 cases. It also keeps a floor of ≈ 0.1–0.3 at
  other frequencies, mostly broadband stripes at speech onsets.
* `examples_freq.png` — one utterance, 4 bands. The |X_fake − X_real| column
  shows why: in this STFT (rectangular 644-sample window) the vocoded band
  leaks across all frequencies at onsets.
* `mask_profiles_time.png` — mask averaged over frequency. For clips ≥ 5 s the
  new mask is ≈ 0 before 3 s and rises at exactly 3 s. For clips < 3 s, where
  nothing was swapped, it stays ≈ 0 everywhere. So it reacts to the vocoded
  speech, not to the position. (The original U-Net is flat, slightly *lower*
  in 3–5 s.)
* `examples_time.png` — the new mask marks low-frequency harmonics inside
  3–5 s, plus a few small patches earlier.
* `training_curves.png` — loss terms per epoch.

## Take-aways

1. **Each U-Net is only good under the fill it was trained with.** The original
   gets 1.00 / 1.00 (freq) with zero fill and 0.26 / 0.00 with real fill. The
   new one gets 1.00 / 0.99 with real fill and 1.00 / 0.02 with zero fill.
   Fidelity/faithfulness say as much about the masking operation as about the
   mask.
2. **Real fill makes the masks localize the manipulation.** freq AUC 0.50 →
   0.74, between 0.70 and 0.78 in every band. The original is ≤ 0.55 except at
   0–1 kHz, where its fixed low-frequency mask happens to sit. time AUC 0.37
   (anti-localized) → 0.68.
3. **Real fill makes the ground truth a sensible reference for time.** The
   ground-truth time mask scores 0.999 / 0.935 with real fill, and the new
   U-Net matches it.
4. **For freq, the ground truth is still not an upper bound.** Refilling the
   nominal 1 kHz band from the real clip flips only 14 % of decisions. The
   detector still sees vocoder traces that leak outside the band in this STFT.
   The new U-Net learns to cover those too (onset stripes, 6–8 kHz floor), so
   it beats the ground truth on faithfulness (0.99 vs 0.14) at the cost of AUC
   (0.74).

## Caveats

* Real fill needs a time-aligned bona fide twin of each fake. That exists only
  for the swap datasets, not for the `realistic` (MLAAD) setup.
* freq training had not converged at epoch 40 (`l_out` and L1 still falling;
  the original freq U-Net trained 74 epochs). It can be continued with
  `--resume --epochs N`.
* AUC is a ranking score. For time clips < 3 s, the new U-Net's mask is ≈ 0
  everywhere (mean 0.0119 inside 3–5 s, 0.0118 outside), yet its AUC against
  the 3–5 s box is 0.68: tiny differences still rank. AUC for clips ≥ 5 s is
  also 0.68.
* The `compute_stft` crop (Nyquist bin and last frame re-padded with zeros,
  see `iSTFT_check/`) is unchanged and applies to every input here, the bona
  fide reference included.

## Files

| file | what it does |
|---|---|
| `common.py` | file lists, bona fide path, dataset, the two fills, detector helper |
| `real_fill_loss.py` | `RealFillLMACLoss`: `losses.LMACLoss` with real (or zero) fill |
| `train_mask.py` | `train_mask.py` with real fill; resumable |
| `recalibrate_bn.py` | recompute BatchNorm averages of a checkpoint (weights unchanged) |
| `evaluate.py` | fidelity / faithfulness (both fills) and AUC for the 3 masks, sharded |
| `figures.py` | the figures above |
| `sanity_check.py` | loss equivalence and the "why" table |
| `run_all.sh` | the whole pipeline in order |

Outputs: `runs/<setup>_realfill/` (checkpoints, `loss_terms.txt`, `train.log`),
`results/<setup>/` (`per_file.csv` with every logit, `summary.txt`,
`summary.csv`, `summary_per_band.csv`), `results/figures/`, `results/logs/`.
Final checkpoints:

* `runs/freq-swapped_realfill/UNet_MP_epoch_40_loss_0.4558_bn_recalibrated.pth`
* `runs/time-swapped_realfill/UNet_MP_epoch_150_loss_0.4779_bn_recalibrated.pth`

Run everything from the repo root with system `python3` (see `run_all.sh`).
