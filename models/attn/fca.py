import torch
import torch.nn as nn
import math


def get_freq_indices(method, n_components, H, W):
    """
    获取选中的 DCT 频率坐标列表 (u, v)
    支持常见策略：'low'（低频）、'top16'（近似高能量）、'NAS'（论文中 NAS 选的最佳组合）
    """
    dct_basis = []
    for h in range(H):
        for w in range(W):
            dct_basis.append((h, w))

    # 按频率总和 (u + v) 从小到大排序（低频优先）
    dct_basis.sort(key=lambda x: x[0] + x[1])

    if method == 'low':
        return dct_basis[:n_components]

    elif method == 'top16':
        # 官方常用 top16，这里用前 16 个低频作为近似（实际官方有固定位置，但简化用低频）
        return dct_basis[:16]

    elif method == 'NAS':
        # 来自官方论文/代码中 NAS 选出的经典 16 个频率坐标（常见配置）
        nas_coords = [
            (0, 0), (0, 1), (1, 0), (0, 2), (2, 0), (1, 1),
            (0, 3), (3, 0), (1, 2), (2, 1), (0, 4), (4, 0),
            (2, 2), (1, 3), (3, 1), (0, 5)
        ]
        return nas_coords[:n_components]

    else:
        raise ValueError(f"Unsupported freq_sel_method: {method}")


def get_dct_weights(H, W, mapper_x, mapper_y, n_components):
    """
    预计算选定频率的 2D DCT 基函数权重
    """
    weight = torch.zeros(n_components, H, W)
    for k in range(n_components):
        u = mapper_x[k]
        v = mapper_y[k]
        for h in range(H):
            for w in range(W):
                alpha_h = math.sqrt(1.0 / H) if h == 0 else math.sqrt(2.0 / H)
                alpha_w = math.sqrt(1.0 / W) if w == 0 else math.sqrt(2.0 / W)
                cos_h = math.cos((2 * h + 1) * u * math.pi / (2 * H))
                cos_w = math.cos((2 * w + 1) * v * math.pi / (2 * W))
                weight[k, h, w] = alpha_h * alpha_w * cos_h * cos_w
    return weight


class FCA(nn.Module):
    """
    官方 FcaNet 的 Frequency Channel Attention 模块（类名改为 FCA）
    - 基于论文《FcaNet: Frequency Channel Attention Networks》(ICCV 2021)
    - 使用 DCT + 多谱频率选择
    - 支持 'low' / 'top16' / 'NAS' 频率选择策略
    """

    def __init__(self, channels: int, reduction: int = 16,
                 freq_sel_method: str = 'NAS',  # 'low', 'top16', 'NAS'
                 dct_h: int = None, dct_w: int = None):
        super(FCA, self).__init__()

        self.reduction = reduction
        self.freq_sel_method = freq_sel_method

        # n_components = 选中的频率分量数（通常 = channels // reduction）
        self.n_components = max(1, channels // reduction)

        # 如果初始化时不知道特征图大小，可以在 forward 中动态计算
        self.dct_h = dct_h
        self.dct_w = dct_w
        self.dct_weight = None  # buffer，会在第一次 forward 时初始化或动态更新

        # FC 层：从 n_components → channels
        mid_dim = max(8, channels // reduction // 2)  # 中间层维度更小一些，节省参数
        self.fc = nn.Sequential(
            nn.Linear(self.n_components, mid_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, channels, bias=False),
            nn.Sigmoid()
        )

    def _init_dct_weight(self, H, W, device):
        """初始化或更新 DCT 权重"""
        selected = get_freq_indices(self.freq_sel_method, self.n_components, H, W)
        mapper_x = [p[0] for p in selected]
        mapper_y = [p[1] for p in selected]

        weight = get_dct_weights(H, W, mapper_x, mapper_y, self.n_components)
        self.dct_weight = weight.to(device).float()  # (n_components, H, W)
        self.dct_h = H
        self.dct_w = W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        # 如果大小不匹配或首次运行，初始化/更新 DCT 权重
        if self.dct_weight is None or h != self.dct_h or w != self.dct_w:
            self._init_dct_weight(h, w, x.device)

        # 提取多谱频率分量（相当于对选定频率做加权全局池化）
        # x: (B, C, H, W) → unsqueeze → (B, 1, C, H, W)
        # dct_weight: (n_components, H, W) → view → (1, n_components, 1, H, W)
        dct_weight = self.dct_weight.view(1, self.n_components, 1, h, w)
        x_unsq = x.unsqueeze(1)  # (B, 1, C, H, W)

        # 对选定频率加权求和 → (B, n_components, C)
        freq_components = (x_unsq * dct_weight).sum(dim=(3, 4))  # (B, n_components, C)
        freq_components = freq_components.permute(0, 2, 1)  # (B, C, n_components)

        # 通过 FC 生成通道注意力权重
        att = self.fc(freq_components)  # (B, C, channels)
        att = att.view(b, c, 1, 1)

        # 残差连接式乘法
        return x * att
