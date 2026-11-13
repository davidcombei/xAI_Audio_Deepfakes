"""How a mask is applied to a spectrogram: zero fill, real fill or average fill.

The original pipeline removes the masked-out part of the spectrogram by
multiplying it with zero. With real fill the masked-out part is filled with the
same time-frequency bins of the *bona fide clip the fake was made from*:

    zero fill:  keep   = m * X_fake
                remove = (1 - m) * X_fake

    real fill:  keep   = m * X_fake + (1 - m) * X_real
                remove = (1 - m) * X_fake + m * X_real

X is the complex STFT (magnitude and phase), m the mask; all three have the same
size, (batch, 513, 313). With a 0/1 mask, "keep" is the fake inside the mask and
the real clip outside it, and "remove" is the other way round. Outside the
manipulated region X_fake == X_real, so the mask value there does not change
either input.

Real fill needs a bona fide clip (``Setup.real_dir``).

Realistic (MLAAD) setup: the bona fide clip is the M-AILABS clip the fake was
generated from. It is a different voice, and often a different length or language,
so it is not time-aligned with the fake and X_fake != X_real everywhere. There,
real fill splices two different recordings.

Average fill ("avg"): the masked-out part is filled with the setup's average
spectrogram instead of a bona fide clip:

    avg fill:   keep   = m * X_fake + (1 - m) * X_avg
                remove = (1 - m) * X_fake + m * X_avg

The average is taken over the magnitude spectrograms |STFT| of every clip in the
setup's own training metadata (Setup.filelist), one (513, 313) average per setup,
made by compute_avg_spectrogram.py. Each clip is first turned into its spectrogram
and only then averaged; the waveforms are never averaged. The average has no phase
of its own (averaging complex STFTs would cancel out to almost zero), so X_avg uses
the average magnitude with the fake clip's own phase:

    X_avg = avg_magnitude * exp(i * phase(X_fake))

Average real fill ("avgreal"): the masked-out part is filled with one fixed complex
spectrogram per setup, the same for every clip. Its magnitude is the average |STFT| of
the setup's unique bona fide clips (the real_dir counterparts of Setup.filelist), and its
phase comes from Griffin-Lim on that average, so no fake phase is left in the filled bins:

    avgreal fill:   keep   = m * X_fake + (1 - m) * X_avgreal
                    remove = (1 - m) * X_fake + m * X_avgreal

Made by compute_avg_spectrogram.py --source real.

Optimized fill ("optimized"): like avgreal, one fixed complex spectrogram per setup, but the
waveform behind it was optimised through the setup's frozen detector so that the detector scores
it 0 (neither fake nor bona fide), alone and when it replaces part of a clip:

    optimized fill: keep   = m * X_fake + (1 - m) * X_opt
                    remove = (1 - m) * X_fake + m * X_opt

Made by learn_neutral_fill.py --setups <setup> --name neutral_fill_<setup>. Each fill only fits
the detector it was made for.
"""

from __future__ import annotations

import numpy as np
import torch

import config

FILLS = ("real", "zero", "avg", "avgreal", "optimized")

AVG_SPECTROGRAM_DIR = config.PROJECT_ROOT / "output" / "avg_spectrogram"
NEUTRAL_FILL_DIR = config.PROJECT_ROOT / "output" / "neutral_fill"


def avg_spectrogram_path(setup):
    """output/avg_spectrogram/<setup name>.npy, the (513, 313) average magnitude of that setup."""
    return AVG_SPECTROGRAM_DIR / f"{setup.name}.npy"


def avg_real_fill_path(setup):
    return AVG_SPECTROGRAM_DIR / f"{setup.name}_real_fill.npy"


def optimized_fill_path(setup):
    return NEUTRAL_FILL_DIR / f"neutral_fill_{setup.name}.npy"


def load_avg_magnitude(setup, device):
    """The setup's average magnitude spectrogram as a tensor: (513, 313)."""
    avg = np.load(avg_spectrogram_path(setup))
    return torch.tensor(avg, dtype=torch.float32, device=device)


def avg_fill_spectrogram(avg_magnitude, x_fake):
    """Complex spectrogram used by avg fill: the average magnitude with each fake's own phase.

    avg_magnitude: (513, 313) real
    x_fake:        (batch, 513, 313) complex
    returns        (batch, 513, 313) complex
    """
    phase = torch.angle(x_fake)                                  # (batch, 513, 313)
    return avg_magnitude.unsqueeze(0) * torch.exp(1j * phase)    # (batch, 513, 313)


def load_fill_reference(setup, fill, device):
    if fill == "avg":
        return load_avg_magnitude(setup, device)
    if fill == "avgreal":
        return torch.from_numpy(np.load(avg_real_fill_path(setup))).to(device)
    if fill == "optimized":
        return torch.from_numpy(np.load(optimized_fill_path(setup))).to(device)
    return None


def fill_spectrogram(fill, reference, x_fake, x_real):
    """What replaces the masked-out bins, (batch, 513, 313) complex."""
    if fill == "avg":
        return avg_fill_spectrogram(reference, x_fake)
    if fill in ("avgreal", "optimized"):
        return reference.expand_as(x_fake)
    return x_real


def check_fill(setup, fill):
    """Stop early with a clear message if the setup cannot use this fill."""
    if fill not in FILLS:
        raise ValueError(f"unknown fill {fill!r}. Choose one of: {', '.join(FILLS)}")
    if fill == "real" and setup.real_dir is None:
        raise ValueError(
            f"setup {setup.name!r} has no bona fide reference clips (real_dir is None), "
            "so it cannot use real fill. Use fill='zero'."
        )
    if fill == "avg" and not avg_spectrogram_path(setup).exists():
        raise FileNotFoundError(
            f"no average spectrogram for setup {setup.name!r} at {avg_spectrogram_path(setup)}. "
            f"Make it first: python compute_avg_spectrogram.py -s {setup.name}"
        )
    if fill == "avgreal":
        if setup.real_dir is None:
            raise ValueError(f"setup {setup.name!r} has no bona fide clips (real_dir is None), so no avgreal fill")
        if not avg_real_fill_path(setup).exists():
            raise FileNotFoundError(
                f"no average real fill for setup {setup.name!r} at {avg_real_fill_path(setup)}. "
                f"Make it first: python compute_avg_spectrogram.py -s {setup.name} --source real"
            )
    if fill == "optimized" and not optimized_fill_path(setup).exists():
        raise FileNotFoundError(
            f"no optimized fill for setup {setup.name!r} at {optimized_fill_path(setup)}. Make it first: "
            f"python learn_neutral_fill.py --setups {setup.name} --name neutral_fill_{setup.name}"
        )


def keep_masked_part(mask, x_fake, x_real, fill):
    """LMAC "relevant" input: the fake inside the mask; zeros, the real clip or X_avg outside.

    For avg/avgreal/optimized fill, pass fill_spectrogram(...) as x_real.
    """
    if fill == "zero":
        return mask * x_fake
    return mask * x_fake + (1 - mask) * x_real


def remove_masked_part(mask, x_fake, x_real, fill):
    """LMAC "irrelevant" input: the fake outside the mask; zeros, the real clip or X_avg inside.

    For avg/avgreal/optimized fill, pass fill_spectrogram(...) as x_real.
    """
    if fill == "zero":
        return (1 - mask) * x_fake
    return (1 - mask) * x_fake + mask * x_real
