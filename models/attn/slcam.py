import torch
import torch.nn as nn
from .se import SE
from .coord_att import CoordAtt

class SLCAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.se = SE(channels, reduction=reduction)
        self.coord = CoordAtt(channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.spatial(x)
        y = self.se(x)
        y = self.coord(y)
        return y * s
