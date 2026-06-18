from __future__ import annotations

import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

import config



def read_lines(path, parse=lambda s: s) -> list:
    with open(path, "r") as f:
        return [parse(line.strip()) for line in f if line.strip()]


def read_paths(path) -> list[str]:

    return [line.rsplit(",", 1)[0].strip() for line in read_lines(path)]


def read_labeled(path) -> tuple[list[str], list[int]]:
    paths, labels = [], []
    for line in read_lines(path):
        p, label = line.rsplit(",", 1)
        paths.append(p.strip())
        labels.append(int(label.strip()))
    return paths, labels



class AudioDataset(Dataset):


    def __init__(self, paths, audio_processor, root=None, device=None):
        self.paths = list(paths)
        self.ap = audio_processor
        self.root = Path(root) if root is not None else None
        self.device = config.get_device(device) if not isinstance(device, torch.device) else device

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        full = self.root / path if self.root is not None else path
        waveform, _ = self.ap.load_audio(full)
        return waveform.to(self.device), path



def collate_features(batch, audio_processor):
    waveforms, paths = zip(*batch)
    waveforms = torch.stack(waveforms, dim=0)
    _, magnitude, phase = audio_processor.compute_stft(waveforms, has_batch_dim=True)
    features = audio_processor.extract_features(waveforms)
    return features, magnitude, phase, paths


def collate_with_logits(batch, audio_processor, logreg):
    waveforms, _ = zip(*batch)
    waveforms = torch.stack(waveforms, dim=0)
    _, magnitude, phase = audio_processor.compute_stft(waveforms, has_batch_dim=True)
    features = audio_processor.extract_features(waveforms)
    yhat_logits, _ = logreg(torch.mean(features, dim=1))
    return features, magnitude, phase, yhat_logits



def find_wavs(root_dir, max_files: int | None = None) -> list[str]:
    out = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".wav"):
                out.append(os.path.join(dirpath, f))
                if max_files and len(out) >= max_files:
                    return out
    return out


def find_wavs_per_system(root_dir, samples_per_system: int = 3, seed: int = 42):
    rng = random.Random(seed)
    fake_root = os.path.join(root_dir, "fake")
    system_to_paths = defaultdict(list)
    for lang in os.listdir(fake_root):
        lang_dir = os.path.join(fake_root, lang)
        if not os.path.isdir(lang_dir):
            continue
        for system in os.listdir(lang_dir):
            system_dir = os.path.join(lang_dir, system)
            if not os.path.isdir(system_dir):
                continue
            for dirpath, _, filenames in os.walk(system_dir):
                for f in filenames:
                    if f.endswith(".wav"):
                        system_to_paths[system].append((os.path.join(dirpath, f), lang))

    results = []
    for system, paths in system_to_paths.items():
        for path, lang in rng.sample(paths, min(samples_per_system, len(paths))):
            results.append((path, system, lang))
    return results


def find_wavs_per_language_and_speaker(
    root_dir, samples_per_language: int = 6, samples_per_speaker: int = 3, seed: int = 42
):
    rng = random.Random(seed)
    all_results = []
    for lang1 in os.listdir(root_dir):
        lang1_dir = os.path.join(root_dir, lang1)
        if not os.path.isdir(lang1_dir):
            continue
        speaker_pool = []
        for lang2 in os.listdir(lang1_dir):
            by_book_dir = os.path.join(lang1_dir, lang2, "by_book")
            if not os.path.isdir(by_book_dir):
                continue
            for gender in os.listdir(by_book_dir):
                gender_dir = os.path.join(by_book_dir, gender)
                if not os.path.isdir(gender_dir):
                    continue
                for speaker in os.listdir(gender_dir):
                    speaker_dir = os.path.join(gender_dir, speaker)
                    if not os.path.isdir(speaker_dir):
                        continue
                    for book in os.listdir(speaker_dir):
                        wavs_dir = os.path.join(speaker_dir, book, "wavs")
                        if not os.path.isdir(wavs_dir):
                            continue
                        wavs = [
                            os.path.join(wavs_dir, f)
                            for f in os.listdir(wavs_dir)
                            if f.endswith(".wav")
                        ]
                        if wavs:
                            speaker_pool.append(
                                (speaker, rng.sample(wavs, min(samples_per_speaker, len(wavs))))
                            )

        selected = []
        rng.shuffle(speaker_pool)
        for speaker, wavs in speaker_pool:
            remaining = samples_per_language - len(selected)
            if remaining <= 0:
                break
            selected.extend((f, speaker, lang1) for f in wavs[:remaining])
        all_results.extend(selected)
    return all_results
