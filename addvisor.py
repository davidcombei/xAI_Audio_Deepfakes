import torch.nn as nn
import torch
from audioprocessor import AudioProcessor
import torch.nn.functional as F
import sys
from torch.autograd import Variable
import math
import numpy as np

audio_processor = AudioProcessor()

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
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
            nn.LeakyReLU(0.2, inplace=True)
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
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: b, 1, 512, 249
        x1 = self.e1(x)   # b, 32, 256, 249
        x2 = self.e2(x1)  # b, 64, 128, 249
        x3 = self.e3(x2)  # b, 128, 64,124
        x4 = self.e4(x3)  #b, 256, 32, 62
        b = self.bottleneck(x4)  # b, 512, 32, 62
        y4 = self.up4(b)                    #b, 256, 64, 124
        y4 = torch.cat([y4, x3], dim=1)     # b, 384, 64, 124
        y4 = self.d4(y4)                    # b, 256, 64, 124
        y3 = self.up3(y4)                   # b, 128, 128, 249
        y3 = torch.cat([y3, x2], dim=1)     # b, 192, 128, 249
        y3 = self.d3(y3)                    # b, 128, 128, 249
        y2 = self.up2(y3)                   # b, 64, 256, 249
        y2 = torch.cat([y2, x1], dim=1)     # b, 96, 256, 249
        y2 = self.d2(y2)                    # b, 64, 256, 249
        y1 = self.up1(y2)                   # b, 32, 512, 249
        y1 = torch.cat([y1, x], dim=1)      # b, 33, 512, 249
        y1 = self.d1(y1)                    # b, 32, 512, 249

        mask = self.mask_head(y1)           # b, 1, 512, 249
        return mask