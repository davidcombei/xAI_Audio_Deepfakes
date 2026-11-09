"""Shared helpers used by every script in iSTFT_check/. Read this file first.

Words used in all files of this folder:
  TF           time-frequency. An explanation computed on the spectrogram. The detector
               only hears it after the inverse STFT (iSTFT).
  wf           waveform. An explanation computed directly on the audio samples
               (no STFT, no iSTFT).
  frame        one column of the spectrogram (about 20 ms). A 5 s clip has 248 frames.
  time profile one number per frame, e.g. "how important is each frame".
               A TF explanation becomes a time profile by averaging over frequency.
  mask         a time profile with values between 0 and 1 that we multiply the audio
               with (1 = keep, 0 = remove).
"""

# The main ADDvisor modules (config, audio, detector, ...) live one folder up.
# Add that folder to Python's search path so we can import them.
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import soundfile
import torch
from sklearn.metrics import roc_auc_score

import config
from audio import AudioProcessor
from data import read_paths
from detector import WaveformDetector, build_detector
from metrics import ground_truth_time
from models import TorchLogReg

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
OUT_DIR = config.PROJECT_ROOT / "output" / "istft_check"   # all results of this folder go here

SAMPLE_RATE = 16000     # samples per second
N_SAMPLES = 80000       # samples in a 5 s clip
N_FRAMES = 248          # spectrogram frames in a 5 s clip
HOP_LENGTH = 322        # a new frame starts every 322 samples
WIN_LENGTH = 644        # each frame is 644 samples long (so frames overlap by half)

SWAP_ONSET_SEC = 3.0    # in the time-swapped clips, 3-5 s were replaced by a vocoder (HiFi-GAN)

# Make sure these numbers match the STFT settings of the main pipeline (config.py).
assert SAMPLE_RATE == config.SAMPLING_RATE
assert N_SAMPLES == config.AUDIO_LENGTH * config.SAMPLING_RATE
assert N_FRAMES == config.N_TIME_FRAMES
assert HOP_LENGTH == config.HOP_LENGTH
assert WIN_LENGTH == config.WIN_LENGTH


def file_list(setup, use_all=False, limit=None):
    """Audio files to process: the setup's 100-file evaluation list, or (use_all) every wav in its folder."""
    if use_all:
        paths = []
        for wav_file in sorted(setup.data_dir.glob("*.wav")):
            paths.append(str(wav_file.relative_to(config.PROJECT_ROOT)))
    else:
        paths = read_paths(setup.filelist)
    if limit is not None:
        paths = paths[:limit]
    return paths


# --------------------------------------------------------------------------- #
# Loading the models
# --------------------------------------------------------------------------- #
def use_stable_attention():
    """Use PyTorch's plain ("math") attention, like train_mask.py does.
    It keeps gradients through wav2vec2 numerically stable."""
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def load_audio_processor(device):
    """The main pipeline's AudioProcessor (STFT, iSTFT, wav2vec2 features), made ~4x faster.

    The detector only reads wav2vec2's hidden_states[9], the input of layer 9.
    So we keep layers 0..9 and throw the rest away: the features stay exactly
    the same. (The checkpoint only has trained weights for the first 9 layers
    anyway; its config says 48 layers, the others are random.)
    """
    audio_processor = AudioProcessor(device=device)
    wav2vec2 = audio_processor.wav2vec2              # loads the model
    n_layers_to_keep = config.W2V2_HIDDEN_LAYER + 1
    wav2vec2.encoder.layers = wav2vec2.encoder.layers[:n_layers_to_keep]
    return audio_processor


def load_tf_detector(setup, audio_processor, device):
    """The detector as the main pipeline uses it.

    Input: a spectrogram (magnitude, phase). It turns it back into audio with
    the iSTFT, then runs wav2vec2 + logistic regression.
    Output: one logit per clip (> 0 means "fake").
    """
    return build_detector(setup, audio_processor, device=device)


def load_waveform_detector(setup, audio_processor, device):
    """The same detector (same logistic regression), but its input is the waveform itself.

    Input: waveform (batch, 80000). Output: logits (batch, 1). No STFT, no iSTFT.
    """
    logistic_regression = TorchLogReg(setup.detector_ckpt).to(device)
    detector = WaveformDetector(audio_processor, logistic_regression).to(device)
    detector.eval()
    return detector


# --------------------------------------------------------------------------- #
# Frames <-> samples
#
# Frame t is centred on sample t*322 and covers samples [t*322 - 322, t*322 + 322).
# So frames overlap by half, and every sample lies inside exactly two frames.
# --------------------------------------------------------------------------- #
def frames_to_samples(frame_mask):
    """Turn one mask value per frame into one mask value per audio sample.

    Input:  (batch, 248)    one value per frame
    Output: (batch, 80000)  one value per sample

    The 322 samples between the centres of frame k and frame k+1 lie inside
    exactly those two frames, so they get the average of the two values. This
    is what happens when you scale whole spectrogram frames and then run the
    iSTFT, so masking the waveform with this is the same as masking the
    spectrogram frame by frame, minus the iSTFT.
    """
    pieces = []
    for k in range(N_FRAMES):
        if k + 1 < N_FRAMES:
            next_value = frame_mask[:, k + 1]
        else:
            next_value = frame_mask[:, k]          # after the last frame: keep the last value
        value = (frame_mask[:, k] + next_value) / 2                   # (batch,)
        pieces.append(value.unsqueeze(1).repeat(1, HOP_LENGTH))       # (batch, 322)

    # 248 * 322 = 79856 samples so far; the last 144 samples follow the last frame.
    n_left = N_SAMPLES - N_FRAMES * HOP_LENGTH
    last_value = frame_mask[:, N_FRAMES - 1]
    pieces.append(last_value.unsqueeze(1).repeat(1, n_left))          # (batch, 144)
    return torch.cat(pieces, dim=1)                                   # (batch, 80000)


def samples_to_frames(per_sample):
    """Add up a per-sample signal (e.g. a gradient) inside each frame.

    Input:  (batch, 80000)
    Output: (batch, 248)    frame t = sum of the samples in [t*322 - 322, t*322 + 322)
    """
    frame_sums = []
    for t in range(N_FRAMES):
        start = max(0, t * HOP_LENGTH - WIN_LENGTH // 2)      # the first frame starts before the clip
        end = min(N_SAMPLES, t * HOP_LENGTH + WIN_LENGTH // 2)
        frame_sums.append(per_sample[:, start:end].sum(dim=1))
    return torch.stack(frame_sums, dim=1)


# --------------------------------------------------------------------------- #
# Ground truth and the localization score
# --------------------------------------------------------------------------- #
def frame_times():
    """Centre time (seconds) of every frame: an array of 248 numbers."""
    return np.arange(N_FRAMES) * HOP_LENGTH / SAMPLE_RATE


def gt_time_profile(audio_path):
    """Ground truth of a time-swapped clip, one value per frame: 1 = manipulated (3-5 s), 0 = real.

    Same definition as metrics.ground_truth_time in the main pipeline.
    """
    gt_2d = ground_truth_time(audio_path, (1, N_FRAMES))    # shape (1, 248)
    return gt_2d[0].numpy()


def localization_auc(profile, gt):
    """Does the explanation point at the manipulated frames?

    ROC-AUC: 1 = every manipulated frame scores higher than every real one,
    0.5 = chance, below 0.5 = it points at the real part instead.
    """
    return float(roc_auc_score(gt, profile))


def original_duration(audio_path):
    """Length (seconds) of the clean clip that a time swap was made from (nan if not found).

    The swapped clips were padded to 5 s *before* the swap. So if the original
    is shorter than 5 s, part of the "manipulated" 3-5 s is just silence (and
    below 3 s, nothing was manipulated at all).
    """
    clean_clip = config.TRAIN_AUDIO_DIR / Path(audio_path).name
    if not clean_clip.exists():
        return float("nan")
    return float(soundfile.info(str(clean_clip)).duration)
