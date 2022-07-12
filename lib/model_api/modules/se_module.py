import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.squeeze(x)
        x = x.view(x.size(0), -1) 
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)

        return x


class SEConvBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // r, kernel_size=1, stride=1),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels // r, in_channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.squeeze(x)
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class ChannelSqueeze(nn.Module):
    def __init__(self, in_channels, r=4):
        super().__init__()
        self.ch_squeeze = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // r, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels // r),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // r, in_channels // r, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels // r),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        x = self.ch_squeeze(x)
        return x
    
    
class ChannelExcitation(nn.Module):
    def __init__(self, in_channels, r=4):
        super().__init__()
        self.ch_excitation = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * r, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels * r),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * r, in_channels * r, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels * r),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        x = self.ch_excitation(x)
        return x