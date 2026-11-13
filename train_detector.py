from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

import config
from audio import AudioProcessor
from data import read_labeled, read_paths
from models import SpectrogramCNN


def compute_eer(y_true, y_scores) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    return float(brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0))



def _embed(paths, ap, label, desc):
    feats, labels = [], []
    for path in tqdm(paths, desc=desc, ascii=True):
        audio, _ = ap.load_audio(path)
        feature = ap.extract_features(audio) 
        feats.append(feature.mean(0).cpu().numpy())
        labels.append(label)
    return feats, labels


def fit_and_save_logreg(X, y, out_path, C=1.0):
    print(f"total={len(y)}  fake={int((y == 1).sum())}  real={int((y == 0).sum())}")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(random_state=42, C=C, max_iter=10000)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_score = model.predict_proba(X_te)[:, 1]
    print(f"accuracy: {accuracy_score(y_te, y_pred):.4f}  EER: {compute_eer(y_te, y_score) * 100:.4f}%")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    print(f"saved -> {out_path}")


def train_logreg(fake_meta, real_meta, out_path, C=1.0, device=None):
    ap = AudioProcessor(device=device)
    fake_feats, fake_labels = _embed(read_paths(fake_meta), ap, 1, "fake")
    real_feats, real_labels = _embed(read_paths(real_meta), ap, 0, "real")
    X = np.array(fake_feats + real_feats)
    y = np.array(fake_labels + real_labels)
    fit_and_save_logreg(X, y, out_path, C)


class _SpecDataset(Dataset):
    def __init__(self, metadata, ap, root):
        self.paths, self.labels = read_labeled(metadata)
        self.ap = ap
        self.root = Path(root)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        waveform, _ = self.ap.load_audio(self.root / self.paths[idx])
        _, magnitude, _ = self.ap.compute_stft(waveform, has_batch_dim=False, logscale=True)
        return magnitude, self.labels[idx]


def train_cnn(metadata, root=None, out_path=None, epochs=1000, batch_size=32, lr=3e-5, device=None):
    device = config.get_device(device)
    root = root or config.TRAIN_AUDIO_DIR
    out_path = Path(out_path or (config.CKPTS_DIR / "cnn_classifier" / "best_model.pth"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ap = AudioProcessor(device=device)
    dataset = _SpecDataset(metadata, ap, root)
    n_train = int(0.8 * len(dataset))
    train_set, test_set = random_split(
        dataset, [n_train, len(dataset) - n_train],
        generator=torch.Generator().manual_seed(42),
    )

    def collate(batch):
        mags, labels = zip(*batch)
        return torch.stack(mags).unsqueeze(1), torch.tensor(labels, dtype=torch.long)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = SpectrogramCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_eer = 1.0

    for epoch in range(epochs):
        model.train()
        for mags, labels in tqdm(train_loader, desc=f"epoch {epoch + 1}", ascii=True):
            mags, labels = mags.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(mags), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        y_true, y_scores = [], []
        with torch.no_grad():
            for mags, labels in test_loader:
                probs = F.softmax(model(mags.to(device)), dim=1)[:, 1]
                y_true.extend(labels.numpy())
                y_scores.extend(probs.cpu().numpy())
        eer = compute_eer(y_true, y_scores)
        print(f"epoch {epoch + 1}: test EER {eer * 100:.4f}%")
        if eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), out_path)
            print(f"  saved (EER {best_eer * 100:.4f}%) -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="kind", required=True)

    lr = sub.add_parser("logreg", help="logistic regression on wav2vec2 features")
    lr.add_argument("--fake", required=True, help="metadata of fake wavs")
    lr.add_argument("--real", required=True, help="metadata of real wavs")
    lr.add_argument("--out", required=True, help="output .joblib path")
    lr.add_argument("--C", type=float, default=1.0)

    cnn = sub.add_parser("cnn", help="CNN over log-magnitude spectrograms")
    cnn.add_argument("--metadata", required=True, help="path,label metadata")
    cnn.add_argument("--root", default=None, help="root dir for the wav paths")
    cnn.add_argument("--out", default=None)
    cnn.add_argument("--epochs", type=int, default=1000)
    cnn.add_argument("--batch-size", type=int, default=32)
    cnn.add_argument("--lr", type=float, default=3e-5)
    return p.parse_args()


def main():
    args = parse_args()
    if args.kind == "logreg":
        train_logreg(args.fake, args.real, args.out, C=args.C)
    else:
        train_cnn(args.metadata, root=args.root, out_path=args.out,
                  epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
