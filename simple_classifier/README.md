# simple_classifier — ADDvisor explanations of a simpler detector

The ADDvisor detector (wav2vec2 + logistic regression) is replaced by a
**logistic regression on the log-mel spectrogram averaged over time**:

    waveform (5 s, 16 kHz) -> 80-band mel spectrogram (Hann, n_fft 1024, hop 322)
      -> log(mel + 1e-6) -> mean over time (80 numbers) -> standardise -> logistic regression

Everything else is unchanged: the same U-Net, the same LMAC loss, the mask is
applied to the pipeline's 512 × 248 STFT, the masked spectrogram goes through
the ISTFT, and the detector scores the waveform. The U-Net was trained against
the new detector with **zero fill** (original ADDvisor) and with **real fill**
(`new_masking/`: masked-out bins taken from the bona fide clip). Both swap
setups, trained and evaluated on **all** files.

Because the detector is linear in the log-mel values, its logit splits
**exactly** into one contribution per mel band and frame:
`logit(fake) − logit(bona fide) = Σ_k,t w_k/σ_k · (logmel_fake[k,t] − logmel_real[k,t]) / T`
(checked: error ≤ 4e-5). This gives a white-box reference explanation,
**classifier evidence**: the STFT bins that push the clip towards "fake" (each mel band's
contribution shared among the bins of its filter; bins above 0.1 % of the
clip's strongest one).

## Detectors

Fakes vs. their bona fide clips (`LJSpeech_vocoded22K/<id>.wav`), 80 / 20
split by utterance, classes re-weighted (`detectors/*.txt`):

| setup | held-out balanced accuracy | EER | called fake, all fakes | called fake, all bona fide |
|---|---:|---:|---:|---:|
| freq-swapped | 0.891 | 10.8 % | 0.885 | 0.096 |
| time-swapped | 0.962 | 3.5 % | 0.953 | 0.031 |

(wav2vec2 detectors: 1.00 / 0.974 of the fakes called fake.)
`C` = 0.1 / 1 / 10 / 100 gives the same accuracy within 2 points.

What the detectors weigh (`results/figures/detector_weights.png`): both put a
large weight on the **lowest mel band (≈ 16 Hz, i.e. DC / sub-bass)**. The freq
detector also uses a few bands around 4 kHz and 6–7 kHz. The time detector uses almost
nothing but the lowest bands.

## Results (all files)

Fidelity: share of clips where keeping only the masked part keeps the decision.
Faithfulness: mean of p_fake(original) − p_fake(masked part removed).
AUC: per-clip ROC-AUC of the mask against the ground-truth region. Same
definitions as `new_masking/evaluate.py`; every clip is counted, including the
ones the detector calls bona fide.

### time-swapped (5 000 clips)

| mask | Fidelity, zero fill | Faithfulness, zero fill | Fidelity, real fill | Faithfulness, real fill | AUC |
|---|---:|---:|---:|---:|---:|
| U-Net, zero fill | **0.996** | **0.888** | 0.977 | 0.772 | 0.498 |
| U-Net, real fill | 0.953 | 0.850 | **1.000** | **0.889** | 0.465 |
| classifier evidence | 0.953 | −0.044 | 0.997 | 0.054 | 0.571 |
| ground truth | 0.953 | 0.705 | 1.000 | 0.897 | 1.000 |

### freq-swapped (40 000 clips)

| mask | Fidelity, zero fill | Faithfulness, zero fill | Fidelity, real fill | Faithfulness, real fill | AUC |
|---|---:|---:|---:|---:|---:|
| U-Net, zero fill | **0.979** | **0.644** | 0.369 | 0.017 | 0.497 |
| U-Net, real fill | 0.885 | 0.265 | **0.986** | 0.606 | 0.509 |
| classifier evidence | 0.885 | 0.295 | 0.886 | **0.689** | 0.622 |
| ground truth | 0.885 | −0.153 | 0.779 | 0.009 | 1.000 |

Per band (`results/freq-swapped/summary.txt`), real fill, faithfulness:

| band (kHz) | 0–1 | 1–2 | 2–3 | 3–4 | 4–5 | 5–6 | 6–7 | 7–8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net, real fill | 0.894 | 0.777 | 0.415 | 0.396 | 0.461 | 0.726 | 0.895 | 0.289 |
| classifier evidence | 0.000 | 0.872 | 0.695 | 0.704 | 0.742 | 0.892 | 0.968 | 0.641 |
| ground truth | 0.010 | 0.009 | 0.006 | 0.053 | −0.024 | 0.068 | −0.005 | −0.042 |

For comparison, the wav2vec2 detector (`new_masking/README.md`): freq U-Net
real fill 1.000 / 0.990, AUC 0.741; time 0.999 / 0.932, AUC 0.678.

## What the explanations show

* **time-swapped: the U-Nets find the detector's shortcut, a DC offset added
  by the vocoder.** Both masks cover only STFT bins 0–4 (0–78 Hz), and only
  in 3–5 s (mask mean over frequency 0.001 before 3 s, 0.011 after;
  `mask_profiles_time.png`, `examples.png`). Checked on 147 clips ≥ 5 s: the
  vocoded 3–5 s segment has a median **260×** more energy below 80 Hz than the
  bona fide one (in every clip). Its median DC offset is 2.8e-3, against 1.3e-5 for
  the bona fide clip. Refilling those few bins from the bona fide clip flips
  the detector as often as refilling the whole 3–5 s region (faithfulness
  0.889 vs 0.897). The explanation is faithful *and* correct about the model.
  AUC ≈ 0.5 only says that a 0–78 Hz strip does not cover the full-height 3–5 s box.
* **freq-swapped: the evidence is not in the swapped band.** Refilling the
  whole nominal band from the bona fide clip changes the decision almost never
  (ground truth, real fill: faithfulness 0.009 in total, ≤ 0.07 in every band). For 1–2 kHz
  swaps, 89 % of the logit difference fake − bona fide comes from mel bands
  outside 1–2 kHz (computed from the features). The swaps were made by a hard
  rectangular band replacement in an STFT (`vocoder.py`), and the detector
  reacts to the leakage and to the lowest and 6–7 kHz bands. The real-fill U-Net
  marks the **two edges of the swapped band** (where vocoded and bona fide
  content are spliced), a 7 kHz line and sub-80 Hz; the zero-fill U-Net marks
  mostly the 7 kHz line (the detector's largest weight)
  (`mask_profiles_freq.png`, `examples.png`). Again AUC ≈ 0.5 is the expected score for a
  correct explanation of this model.
* **Faithfulness depends on the fill, as with wav2vec2.** Each U-Net is good
  under the fill it was trained with and poor under the other (freq zero-fill
  U-Net: 0.644 zero fill, 0.017 real fill). Zeroing the ground-truth band of a
  *bona fide* clip makes the freq detector call it fake every time
  (`results/logs/sanity_check.txt`).
* **The exact explanation is not automatically the most faithful under STFT
  masking.** For time swaps, refilling every bin where the classifier sees
  "fake" evidence leaves 96 % of the clips fake (faithfulness 0.054). The
  evidence sits in a few low-energy bins: 0.1 % of the bins hold 73 % of it.
  Masking on the pipeline's STFT (rectangular 644-sample window) and analysing
  again with a Hann mel spectrogram after the ISTFT leaks neighbouring content
  into those bins, and the log amplifies it. For freq swaps, where the evidence is
  spread over more bins, it is the most faithful mask under real fill (0.689, best
  in 7 of 8 bands), and the real-fill U-Net does not reach it (0.606).
  So faithfulness measures the explanation *and* how well the masking domain
  matches the detector's.

Take-away: with a model whose reasoning is known, the ADDvisor U-Net recovers
what the model uses, a DC / sub-bass artifact and band-edge or 7 kHz cues,
not the nominal manipulated region. Low AUC against the ground truth is
therefore not evidence of a bad explanation: it can mean the detector does not
look at the manipulation. (Not tested: whether the wav2vec2 time-swap U-Net's
< 1 kHz mask, `iSTFT_check/README.md`, is the same DC artifact.)

## Caveats

* The detectors are weak on freq swaps (89 % balanced accuracy), and 11.5 % of the
  freq fakes are called bona fide. For those clips the masks explain a "bona
  fide" decision; they are still counted in every metric, as in the other experiments.
* The learnable LMAC loss weights shrink towards 0 during training (they are
  minimised along with the loss: freq ends at 0.02 / 0.002 / 0.03). The freq U-Nets'
  `l_out` plateaus at 0.31–0.38 (`training_curves.png`), so the freq U-Nets
  are probably not as good as they could be. Same recipe as the repo's `train_mask.py`.
* BatchNorm statistics of all four final checkpoints were recomputed with
  `new_masking/recalibrate_bn.py` (lesson from `new_masking/`); the
  `*_bn_recalibrated.pth` files are the ones evaluated.
* "Classifier evidence" uses the bona fide clip, so like real fill it exists
  only for the swap setups. The `realistic` (MLAAD) setup was not run (no
  ground truth, no bona fide twins).
* The `compute_stft` crop (Nyquist bin and last frame) applies everywhere. The
  detectors were trained on STFT → ISTFT round trips, so they see clips exactly
  as the explanation pipeline produces them.

## Files

| file | what it does |
|---|---|
| `mel_detector.py` | `MelLogRegDetector`: features, logit, exact contributions, evidence mask |
| `common.py` | paths + detector call; data and fills are reused from `new_masking/common.py` |
| `train_detector.py` | fit the logistic regression per setup (`detectors/`) |
| `sanity_check.py` | detector decisions on masked / zeroed inputs; contributions add up |
| `lmac_loss.py` | the LMAC loss with the mel detector, zero or real fill |
| `train_mask.py` | train the U-Net against the mel detector; resumable |
| `evaluate.py` | fidelity / faithfulness (both fills) and AUC for the 4 masks, sharded |
| `figures.py` | the figures in `results/figures/` |
| `run_all.sh` | the whole pipeline in order |

Final U-Nets (`runs/`): freq 60 epochs, time 300 epochs, batch 2 × 15, Adam 3e-5 / 1e-4.

* `runs/freq-swapped_zerofill/UNet_MP_epoch_60_loss_0.0066_bn_recalibrated.pth`
* `runs/freq-swapped_realfill/UNet_MP_epoch_60_loss_0.0069_bn_recalibrated.pth`
* `runs/time-swapped_zerofill/UNet_MP_epoch_300_loss_0.0322_bn_recalibrated.pth`
* `runs/time-swapped_realfill/UNet_MP_epoch_300_loss_0.0323_bn_recalibrated.pth`

Run everything from the repo root with system `python3` (see `run_all.sh`).
