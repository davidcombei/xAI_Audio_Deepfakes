"""How a mask is applied to a spectrogram: zero fill or real fill.

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
"""

from __future__ import annotations

FILLS = ("real", "zero")


def check_fill(setup, fill):
    """Stop early with a clear message if the setup cannot use this fill."""
    if fill not in FILLS:
        raise ValueError(f"unknown fill {fill!r}. Choose one of: {', '.join(FILLS)}")
    if fill == "real" and setup.real_dir is None:
        raise ValueError(
            f"setup {setup.name!r} has no bona fide reference clips (real_dir is None), "
            "so it cannot use real fill. Use fill='zero'."
        )


def keep_masked_part(mask, x_fake, x_real, fill):
    """LMAC "relevant" input: the fake inside the mask; zeros or the real clip outside."""
    if fill == "zero":
        return mask * x_fake
    return mask * x_fake + (1 - mask) * x_real


def remove_masked_part(mask, x_fake, x_real, fill):
    """LMAC "irrelevant" input: the fake outside the mask; zeros or the real clip inside."""
    if fill == "zero":
        return (1 - mask) * x_fake
    return (1 - mask) * x_fake + mask * x_real
