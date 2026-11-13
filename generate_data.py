from __future__ import annotations

import argparse

import config
from data import read_lines
from vocoder import generate_freq_swaps


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="kind", required=True)

    f = sub.add_parser("freq", help="frequency-band swaps")
    f.add_argument("--wav-dir", required=True, help="directory of raw (clean) wavs")
    f.add_argument("--out-dir", default=str(config.SETUPS["freq-swapped"].data_dir))
    f.add_argument("--filelist", default=str(config.METADATA_DIR / "LJSpeech.txt"))
    f.add_argument("--limit", type=int, default=None, help="cap number of files")
    f.add_argument("--band-width", type=int, default=1000)
    f.add_argument("--f-max", type=int, default=8000)
    return p.parse_args()


def main():
    args = parse_args()
    files = read_lines(args.filelist)
    if args.limit:
        files = files[: args.limit]
    generate_freq_swaps(
        args.wav_dir, args.out_dir, files,
        band_width=args.band_width, f_max=args.f_max,
    )


if __name__ == "__main__":
    main()
