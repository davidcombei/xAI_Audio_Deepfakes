from __future__ import annotations

import argparse

import config
from data import read_lines
from vocoder import generate_freq_swaps, generate_time_swaps


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

    t = sub.add_parser("time", help="temporal swaps")
    t.add_argument("--orig-dir", default=None, help="clean clips (default: LJSpeech_vocoded22K)")
    t.add_argument("--vocoded-dir", default=None, help="vocoded clips (default: LJSpeech_hifigan)")
    t.add_argument("--out-dir", default=None)
    t.add_argument("--filelist", default=str(config.METADATA_DIR / "LJSpeech.txt"))
    t.add_argument("--start", type=float, default=3.0)
    t.add_argument("--end", type=float, default=5.0)
    return p.parse_args()


def main():
    args = parse_args()
    files = read_lines(args.filelist)
    if args.kind == "freq":
        if args.limit:
            files = files[: args.limit]
        generate_freq_swaps(
            args.wav_dir, args.out_dir, files,
            band_width=args.band_width, f_max=args.f_max,
        )
    else:
        generate_time_swaps(
            orig_dir=args.orig_dir, vocoded_dir=args.vocoded_dir, out_dir=args.out_dir,
            file_list=files, start_sec=args.start, end_sec=args.end,
        )


if __name__ == "__main__":
    main()
