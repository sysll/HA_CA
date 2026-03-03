import torch
import torch.nn as nn
import torch.nn.functional as F

class HVCA(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(2 * channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_abs = x.abs()
        v = x_abs.sum(dim=2).amax(dim=2)
        h_ = x_abs.sum(dim=3).amax(dim=2)
        z = torch.cat([v, h_], dim=1)
        a = self.fc2(self.act(self.fc1(z)))
        a = self.gate(a).view(b, c, 1, 1)
        return x * a
