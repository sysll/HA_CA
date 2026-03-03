import torch
import torch.nn as nn
from .hvca import HVCA
from .attn.se import SE
from .attn.eca import ECA
from .attn.coord_att import CoordAtt
from .attn.fca import FCA
from .attn.ortho import Ortho
from .attn.epsa import EPSA
from .attn.slcam import SLCAM


def make_attn(attn_type: str, c: int, reduction: int):
    if attn_type == "none":
        return nn.Identity()
    if attn_type == "hvca":
        return HVCA(c, reduction=reduction)
    if attn_type == "se":
        return SE(c, reduction=reduction)
    if attn_type == "eca":
        return ECA(c)
    if attn_type == "ca":
        return CoordAtt(c, reduction=reduction)
    if attn_type == "fca":
        return FCA(c, reduction=reduction)
    if attn_type == "ortho":
        return Ortho(c, reduction=reduction)
    if attn_type == "epsa":
        return EPSA(c, reduction=reduction)
    if attn_type == "slcam":
        return SLCAM(c, reduction=reduction)
    raise ValueError(attn_type)


# ==================== 新增辅助函数（核心修改） ====================
def _add_attentions_to_features(features: nn.Sequential, attn_type: str, reduction: int = 16):
    """
    通用函数：在 features Sequential 的**每个主要 block 后面**插入注意力模块
    - 自动检测每个 block 的输出通道数（支持 VGG 的 Conv、MobileNet/EfficientNet 的 InvertedResidual/MBConv、DenseNet 的 DenseBlock/Transition）
    - 如果 attn_type == "none" 则不做任何修改
    - 这样就实现了“所有模型所有块后面”都集成注意力
    """
    if attn_type == "none":
        return features

    new_modules = []

    for name, module in features.named_children():
        new_modules.append(module)  # 先添加原 block

        # 自动获取输出通道数（适配所有模型）
        out_c = None
        if hasattr(module, 'out_channels'):  # VGG 的 Conv2d
            out_c = module.out_channels
        elif hasattr(module, 'num_features'):  # DenseNet 的 DenseBlock / BatchNorm
            out_c = module.num_features
        else:
            # MobileNet/EfficientNet 的复杂 block（InvertedResidual、MBConv）
            # 反向查找最后一个 Conv2d
            for m in reversed(list(module.modules())):
                if isinstance(m, nn.Conv2d):
                    out_c = m.out_channels
                    break
                if hasattr(m, 'out_channels') and m.out_channels > 0:
                    out_c = m.out_channels
                    break

        if out_c is not None:
            attn = make_attn(attn_type, out_c, reduction)
            new_modules.append(attn)

    return nn.Sequential(*new_modules)


# ==================== 修改后的包装类（保持原接口） ====================

class VGGWithAttn(nn.Module):
    def __init__(self, backbone: nn.Module, attn_type: str = "none", reduction: int = 16):
        super().__init__()
        self.backbone = backbone
        # 把注意力插入到 features 的每个主要块后面
        self.features = _add_attentions_to_features(backbone.features, attn_type, reduction)

    def forward(self, x):
        x = self.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)
        return x


class MobileNetWithAttn(nn.Module):
    def __init__(self, backbone: nn.Module, attn_type: str = "none", reduction: int = 16):
        super().__init__()
        self.backbone = backbone
        # 把注意力插入到 features 的每个主要块后面
        self.features = _add_attentions_to_features(backbone.features, attn_type, reduction)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)
        return x


class DenseNetWithAttn(nn.Module):
    def __init__(self, backbone: nn.Module, attn_type: str = "none", reduction: int = 16):
        super().__init__()
        self.backbone = backbone
        # 把注意力插入到 features 的每个主要块后面
        self.features = _add_attentions_to_features(backbone.features, attn_type, reduction)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        features = self.features(x)
        x = torch.relu(features)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)
        return x


class EfficientNetWithAttn(nn.Module):
    def __init__(self, backbone: nn.Module, attn_type: str = "none", reduction: int = 16):
        super().__init__()
        self.backbone = backbone
        # 把注意力插入到 features 的每个主要块后面
        self.features = _add_attentions_to_features(backbone.features, attn_type, reduction)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)
        return x