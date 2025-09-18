import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.step = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.step(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()

        self.layer1 = DoubleConv(in_channels, 32)
        self.layer2 = DoubleConv(32, 64)
        self.layer3 = DoubleConv(64, 128)
        self.layer4 = DoubleConv(128, 256)

        self.layer5 = DoubleConv(256 + 128, 128)
        self.layer6 = DoubleConv(128 + 64, 64)
        self.layer7 = DoubleConv(64 + 32, 32)
        self.layer8 = nn.Conv3d(32, out_channels, kernel_size=1)

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(self.maxpool(x1))
        x3 = self.layer3(self.maxpool(x2))
        x4 = self.layer4(self.maxpool(x3))

        # Upsample and concatenate
        x5 = F.interpolate(x4, size=x3.shape[2:], mode='trilinear', align_corners=False)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.layer5(x5)

        x6 = F.interpolate(x5, size=x2.shape[2:], mode='trilinear', align_corners=False)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.layer6(x6)

        x7 = F.interpolate(x6, size=x1.shape[2:], mode='trilinear', align_corners=False)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.layer7(x7)

        out = self.layer8(x7)
        return out
