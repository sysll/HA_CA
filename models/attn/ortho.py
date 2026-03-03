# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from typing import Dict, Optional


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