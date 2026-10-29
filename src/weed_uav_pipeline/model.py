"""Small U-Net using the three vegetation-index channels used by the project."""
import torch
from torch import nn
import torch.nn.functional as F


def vegetation_features(rgb):
    """uint8 RGB image/batch -> normalized float ExG, ExGR, CIVE channels."""
    x = rgb.to(dtype=torch.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    exg = 2.0 * g - r - b
    exgr = 3.0 * g - 2.4 * r - b
    cive = 0.441 * r - 0.811 * g + 0.385 * b + (18.78745 / 255.0)
    return torch.stack((exg, exgr, cive), dim=-1).permute(0, 3, 1, 2)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class WeedUNet(nn.Module):
    """Two-class segmenter: output channel 0 is weed; channel 1 is other."""
    def __init__(self, widths=(16, 32, 64, 128), classes=2):
        super().__init__()
        self.down = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        in_ch = 3
        for width in widths:
            self.down.append(DoubleConv(in_ch, width))
            in_ch = width
        self.bottleneck = DoubleConv(widths[-1], widths[-1] * 2)
        self.up = nn.ModuleList()
        in_ch = widths[-1] * 2
        for width in reversed(widths):
            self.up.append(nn.ConvTranspose2d(in_ch, width, 2, stride=2))
            self.up.append(DoubleConv(width * 2, width))
            in_ch = width
        self.head = nn.Conv2d(widths[0], classes, 1)

    def forward(self, x):
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for i in range(0, len(self.up), 2):
            x = self.up[i](x)
            skip = skips[-1 - i // 2]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.up[i + 1](torch.cat((skip, x), dim=1))
        return self.head(x)

