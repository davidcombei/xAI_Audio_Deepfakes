from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

from config import (
    FEATURE_DIM,
    W2V2_HF_ID,
    W2V2_LOCAL_DIR,
    get_device,
)



def zero_mean_unit_var_norm(input_values: torch.Tensor) -> torch.Tensor:
    """Per-utterance zero-mean/unit-variance normalisation (wav2vec2 input)."""
    mean = input_values.mean(dim=-1, keepdim=True)
    std = input_values.std(dim=-1, keepdim=True)
    return (input_values - mean) / (std + 1e-7)


@lru_cache(maxsize=None)
def _load_wav2vec2_cached(source: str, device_str: str) -> Wav2Vec2Model:
    model = Wav2Vec2Model.from_pretrained(source)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model.to(torch.device(device_str))


def load_wav2vec2(device: torch.device | str | None = None) -> Wav2Vec2Model:

    device = get_device(device) if not isinstance(device, torch.device) else device
    source = str(W2V2_LOCAL_DIR) if W2V2_LOCAL_DIR.exists() else W2V2_HF_ID
    return _load_wav2vec2_cached(source, str(device))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.e1 = ConvBlock(1, 32, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))
        self.e2 = ConvBlock(32, 64, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1))
        self.e3 = ConvBlock(64, 128, stride=(2, 2))
        self.e4 = ConvBlock(128, 256, stride=(2, 2))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=(2, 2))
        self.d4 = ConvBlock(384, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(2, 2))
        self.d3 = ConvBlock(192, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=(2, 1), stride=(2, 1))
        self.d2 = ConvBlock(96, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=(2, 1), stride=(2, 1))
        self.d1 = ConvBlock(33, 32)

        self.mask_head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, 1, 512, 248)
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        b = self.bottleneck(x4)
        y4 = self.d4(torch.cat([self.up4(b), x3], dim=1))
        y3 = self.d3(torch.cat([self.up3(y4), x2], dim=1))
        y2 = self.d2(torch.cat([self.up2(y3), x1], dim=1))
        y1 = self.d1(torch.cat([self.up1(y2), x], dim=1))
        return self.mask_head(y1)


def load_mask_predictor(
    checkpoint_path: str | Path, device: torch.device | str | None = None
) -> UNet:

    device = get_device(device) if not isinstance(device, torch.device) else device
    model = UNet().to(device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if any(k.startswith("module.") for k in checkpoint):
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.eval()
    return model



class TorchLogReg(nn.Module):


    def __init__(self, classifier_path: str | Path):
        super().__init__()
        classifier = joblib.load(str(classifier_path))
        self.linear = nn.Linear(FEATURE_DIM, 1)
        self.linear.weight = nn.Parameter(
            torch.tensor(classifier.coef_, dtype=torch.float32), requires_grad=False
        )
        self.linear.bias = nn.Parameter(
            torch.tensor(classifier.intercept_, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x):
        logits = self.linear(x)
        probs = torch.sigmoid(logits)
        return logits, probs


class SpectrogramCNN(nn.Module):


    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 32 * 15, 1024)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        return self.fc4(x)
