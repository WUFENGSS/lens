import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = ResidualConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.block(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ResidualConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class MRCFN(nn.Module):
    """Multi-scale Residual Cost Filtering Network (MR-CFN).

    Input: [cost volume, raw score map]
    Output: refined anomaly logits.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        c1 = max(16, hidden_channels)
        c2 = c1 * 2
        c3 = c1 * 4

        self.enc1 = ResidualConvBlock(in_channels, c1)
        self.enc2 = DownBlock(c1, c2)
        self.enc3 = DownBlock(c2, c3)

        self.bottleneck = DownBlock(c3, c3)

        self.dec3 = UpBlock(c3, c3, c3)
        self.dec2 = UpBlock(c3, c2, c2)
        self.dec1 = UpBlock(c2, c1, c1)

        # Multi-scale residual refinement heads.
        self.head_s1 = nn.Conv2d(c1, 1, kernel_size=1)
        self.head_s2 = nn.Conv2d(c2, 1, kernel_size=1)
        self.head_s3 = nn.Conv2d(c3, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        logit_s1 = self.head_s1(d1)
        logit_s2 = F.interpolate(self.head_s2(d2), size=d1.shape[-2:], mode="bilinear", align_corners=False)
        logit_s3 = F.interpolate(self.head_s3(d3), size=d1.shape[-2:], mode="bilinear", align_corners=False)

        # Residual multi-scale fusion.
        logits = logit_s1 + 0.5 * logit_s2 + 0.25 * logit_s3
        return logits


class CostFilterLite(MRCFN):
    """Backward-compatible alias for previous integration code."""

    pass
