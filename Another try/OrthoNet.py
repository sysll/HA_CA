# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from typing import Dict, Optional

__all__ = ['OrthoNet', 'orthonet18', 'orthonet34', 'orthonet50', 'orthonet101', 'orthonet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class GramSchmidtTransform(nn.Module):
    """
    使用 Gram-Schmidt 正交化生成的固定滤波器进行全局特征压缩
    输入 [bs, c, h, w] → 输出 [bs, c, 1, 1]
    """
    instance: Dict[tuple, Optional['GramSchmidtTransform']] = {}

    @staticmethod
    def build(c: int, h: int):
        key = (c, h)
        if key not in GramSchmidtTransform.instance:
            GramSchmidtTransform.instance[key] = GramSchmidtTransform(c, h)
        return GramSchmidtTransform.instance[key]

    def __init__(self, c: int, h: int):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            filters = self._initialize_orthogonal_filters(c, h, h)
            filters = filters.view(c, h, h)
        self.register_buffer("constant_filter", filters.to(self.device).detach())

    def _initialize_orthogonal_filters(self, c, h, w):
        if h * w < c:
            n = c // (h * w)
            grams = []
            for _ in range(n):
                rand_mat = torch.rand(h * w, 1, h, w)
                grams.append(self._gram_schmidt(rand_mat))
            return torch.cat(grams, dim=0)
        else:
            rand_mat = torch.rand(c, 1, h, w)
            return self._gram_schmidt(rand_mat)

    def _gram_schmidt(self, input):
        def projection(u, v):
            return (v * u).sum() / (u * u).sum() * u

        output = []
        for x in input:
            for y in output:
                x = x - projection(y, x)
            norm = x.norm(p=2)
            if norm > 1e-6:
                x = x / norm
            output.append(x)
        return torch.stack(output)

    def forward(self, x):
        _, _, hi, wi = x.shape
        _, H, W = self.constant_filter.shape
        if hi != H or wi != W:
            x = nn.functional.adaptive_avg_pool2d(x, (H, W))
        # 核心：用正交滤波器做全局加权求和
        return (self.constant_filter * x).sum(dim=(-1, -2), keepdim=True)


class OrthoAttention(nn.Module):
    """简单包装 GramSchmidtTransform，使其更易调用"""
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, fwt: GramSchmidtTransform, input: torch.Tensor):
        while input.size(-1) > 1:  # 多次压缩直到 1×1
            input = fwt(input.to(self.device))
        b = input.size(0)
        return input.view(b, -1)  # [bs, c]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, height, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        # Ortho + SE 注意力
        self.ortho_attn = OrthoAttention()
        self.fca = GramSchmidtTransform.build(planes, height)

        reduction = 16
        self.se = nn.Sequential(
            nn.Linear(planes, planes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(planes // reduction, planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Ortho 压缩 → [bs, c]
        compressed = self.ortho_attn(self.fca, out)
        # SE 生成权重 → [bs, c] → [bs, c, 1, 1]
        excitation = self.se(compressed).unsqueeze(-1).unsqueeze(-1)

        out = out * excitation
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, height, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # Ortho + SE
        self.ortho_attn = OrthoAttention()
        self.fca = GramSchmidtTransform.build(planes * 4, height)

        reduction = 16
        self.se = nn.Sequential(
            nn.Linear(planes * 4, planes * 4 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(planes * 4 // reduction, planes * 4, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        compressed = self.ortho_attn(self.fca, out)
        excitation = self.se(compressed).unsqueeze(-1).unsqueeze(-1)

        out = out * excitation
        out += residual
        out = self.relu(out)

        return out


class OrthoNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # height 参数控制 Gram-Schmidt 滤波器大小，通常随层变小
        self.layer1 = self._make_layer(block, 64,  64, layers[0])
        self.layer2 = self._make_layer(block, 128, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, 16, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, 8,  layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block, planes, height, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, height, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, height))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ----------------- 模型构建函数 -----------------
def orthonet18(num_classes=1000):
    return OrthoNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def orthonet34(num_classes=1000):
    return OrthoNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def orthonet50(num_classes=1000):
    return OrthoNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def orthonet101(num_classes=1000):
    return OrthoNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def orthonet152(num_classes=1000):
    return OrthoNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)


# ----------------- 简单测试 -----------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试 ResNet-18 风格
    model = orthonet18(num_classes=1000).to(device)
    model.eval()

    # 随机输入 (batch=2, 3, 224, 224)
    x = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        out = model(x)
    print("OrthoNet-18 output shape:", out.shape)   # 应为 [2, 1000]

    print("Test passed.")