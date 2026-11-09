"""Make the folder of bona fide clips for real fill on the realistic (MLAAD) setup.

Every MLAAD fake was generated from one M-AILABS clip, and both have the same
file name. The M-AILABS clips sit in many nested folders, so this script makes
one flat folder of links:

    mailabs_real/<name>.wav -> <M-AILABS folder>/.../wavs/<name>.wav

Run once from anywhere:
    python3 link_mailabs_reals.py
"""

import os

import config

MAILABS_DIR = "/mnt/QNAP/comdav/DATA/DATA/m-ailabs"


def find_all_mailabs_clips():
    """Return a dictionary: file name -> full path of that M-AILABS clip."""
    clips = {}
    for folder, _, file_names in os.walk(MAILABS_DIR):
        for file_name in file_names:
            if not file_name.endswith(".wav"):
                continue
            if file_name in clips:
                raise ValueError(f"{file_name} is in M-AILABS twice, the name does not identify the clip")
            clips[file_name] = os.path.join(folder, file_name)
    return clips


def main():
    setup = config.get_setup("realistic")
    clips = find_all_mailabs_clips()
    print(f"{len(clips)} clips found in M-AILABS")

    os.makedirs(setup.real_dir, exist_ok=True)
    n_linked = 0
    n_missing = 0
    for file_name in sorted(os.listdir(setup.data_dir)):
        if not file_name.endswith(".wav"):
            continue
        if file_name not in clips:
            print(f"no M-AILABS clip for {file_name}")
            n_missing += 1
            continue
        link_path = os.path.join(setup.real_dir, file_name)
        if not os.path.islink(link_path):
            os.symlink(clips[file_name], link_path)
        n_linked += 1
    print(f"{n_linked} links in {setup.real_dir}, {n_missing} fakes without a real clip")


if __name__ == "__main__":
    main()
