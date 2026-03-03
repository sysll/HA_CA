# HV-CA: Horizontal and Vertical Pooling Channel Attention

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 1.7](https://img.shields.io/badge/pytorch-1.7-orange.svg)](https://pytorch.org/get-started/previous-versions/)

Official PyTorch implementation of **HV-CA: A Plug-and-Play Channel Attention Driven by Horizontal and Vertical Pooling for Convolutional Neural Networks** (Zhu et al.).

This repository contains the code for the HV-Pooling module and its integration into popular CNN architectures (ResNet, Inception, SwinUnet) for image classification and segmentation tasks.

## 📖 Overview

Channel attention mechanisms typically use **Global Average Pooling (GAP)** to squeeze spatial dimensions into a single statistic per channel. However, GAP discards spatial structure and dilutes salient information — especially in deep layers where feature maps are often sparse.

**HV-CA** replaces GAP with **Horizontal and Vertical Pooling (HV-Pooling)**, which captures the most significant row-wise and column-wise features using induced matrix norms. This preserves both **magnitude** and **positional cues**, resulting in more discriminative channel weights.

## 🔬 HV-Pooling Method

For an input feature map $U \in \mathbb{R}^{H \times W}$ (per channel), traditional global average pooling computes:

$$
z = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} u_{ij}
$$

This treats every spatial location equally and is sensitive to sparsity.

We propose **Horizontal and Vertical Pooling (HV-Pooling)** instead:

$$
\begin{bmatrix}
z_1 \\
z_2
\end{bmatrix}
=
\begin{bmatrix}
\max_j \sum_i u_{ij} \\
\max_i \sum_j u_{ij}
\end{bmatrix}
$$

(where $u_{ij} \geq 0$ after ReLU)

- $z_1$ : maximum column sum — strongest **vertical** strip  
- $z_2$ : maximum row sum — strongest **horizontal** strip  

This produces a compact 2-D descriptor per channel that is both **intensity-aware** and **spatially sensitive**.

<div align="center">
  <img src="hv_pooling.png" alt="HV-Pooling mechanism" width="600"/>
  <br>
  <em>Figure: HV-Pooling extracts the max column sum (vertical direction) and max row sum (horizontal direction) for each channel.</em>
</div>

## 🛠️ Implementation

### HV-Pooling Core Function

```python
import torch
import torch.nn as nn

def Level_vertical_pooling(x):
    """
    Your implementation variant:
    L_inf → max column sum (vertical strongest)
    L1   → max row sum    (horizontal strongest)
    """
    # Note: abs() is optional after ReLU
    L_inf = torch.max(torch.sum(torch.abs(x), dim=3), dim=2).values.unsqueeze(2)
    L1    = torch.max(torch.sum(torch.abs(x), dim=2), dim=2).values.unsqueeze(2)
    
    feature_cat_vec = torch.cat((L_inf, L1), dim=2).flatten(1)   # (B, 2C)
    return feature_cat_vec

class LVP_ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(LVP_ChannelAttention, self).__init__()
        self.LVP = Level_vertical_pooling
        
        self.fc1   = nn.Linear(2 * in_planes, int(1.5 * in_planes))
        self.relu1 = nn.Mish()
        self.fc2   = nn.Linear(int(1.5 * in_planes), in_planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        tmp = x
        x = self.LVP(x)                     # (B, 2 * in_planes)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)                     # (B, in_planes)
        x = x.unsqueeze(2).unsqueeze(3)     # (B, C, 1, 1)
        x = self.sigmoid(x) * tmp
        return x
