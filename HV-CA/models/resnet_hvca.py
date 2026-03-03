import torch
import torch.nn as nn
from .hvca import HVCA

class ResNetHVCA(nn.Module):
    def __init__(self, backbone: nn.Module, reduction: int = 16):
        super().__init__()
        self.backbone = backbone
        chs = self._infer_stage_channels()
        self.attn1 = HVCA(chs[0], reduction=reduction)
        self.attn2 = HVCA(chs[1], reduction=reduction)
        self.attn3 = HVCA(chs[2], reduction=reduction)
        self.attn4 = HVCA(chs[3], reduction=reduction)

    def _infer_stage_channels(self):
        c1 = self.backbone.layer1[-1].bn2.num_features if hasattr(self.backbone.layer1[-1], "bn2") else self.backbone.layer1[-1].bn3.num_features
        c2 = self.backbone.layer2[-1].bn2.num_features if hasattr(self.backbone.layer2[-1], "bn2") else self.backbone.layer2[-1].bn3.num_features
        c3 = self.backbone.layer3[-1].bn2.num_features if hasattr(self.backbone.layer3[-1], "bn2") else self.backbone.layer3[-1].bn3.num_features
        c4 = self.backbone.layer4[-1].bn2.num_features if hasattr(self.backbone.layer4[-1], "bn2") else self.backbone.layer4[-1].bn3.num_features
        return (c1, c2, c3, c4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.attn1(x)
        x = self.backbone.layer2(x)
        x = self.attn2(x)
        x = self.backbone.layer3(x)
        x = self.attn3(x)
        x = self.backbone.layer4(x)
        x = self.attn4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)
        return x
