[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 1.7](https://img.shields.io/badge/pytorch-1.7-orange.svg)](https://pytorch.org/get-started/previous-versions/)

PyTorch implementation. 
This repository contains the code for the HV-Pooling module and its integration into popular CNN architectures (ResNet, Inception, SwinUnet) for image classification and segmentation.

---

## 📖 Overview

Channel attention mechanisms typically use **global average pooling (GAP)** to squeeze feature maps into a single statistic. However, GAP discards spatial structure and dilutes salient information, especially in sparse feature maps common in deep CNNs.  
**HV-CA** replaces GAP with **Horizontal and Vertical Pooling (HV-Pooling)**, which extracts the most significant row and column features using induced matrix norms. This preserves both magnitude and positional cues, leading to more accurate channel weights.

### 🔬 Method: Horizontal and Vertical Pooling (HV-Pooling)

For a feature map $U \in \mathbb{R}^{H \times W}$, the traditional global average pooling will map it into a statistic $z$:

$$z = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} u_{ij}$$

Here, $u_{ij}$ represents the element in the $i$-th row and $j$-th column. We argue that features obtained through global average pooling struggle to fully capture the significance of a feature map. First, global average pooling lacks positional awareness. Each element $u_{ij}$ contributes equally to the computed statistic $z$, disregarding spatial importance. In contrast, max pooling considers only the most dominant element, preserving its location. Additionally, feature maps are often sparse matrices, meaning most elements $u_{ij}$ are zero. As a result, only a small subset of elements significantly contribute to $z$, while the scaling factor $\frac{1}{H \times W}$ further suppresses these values, weakening their impact.

To address these limitations, we propose **Horizontal and Vertical Pooling (HV-Pooling)** as an alternative to global average pooling. This method enhances the model's ability to capture deep feature information. The pooling formulation is summarized as follows:

$$\left[\begin{array}{c} z_1 \\ z_2 \end{array}\right] = \left[\begin{array}{c} \|U\|_1 \\ \|U\|_\infty \end{array}\right] = \left[\begin{array}{c} \max_j \sum_i |u_{ij}| \\ \max_i \sum_j |u_{ij}| \end{array}\right]$$

We obtain a statistical vector $\mathbf{z} \in \mathbb{R}^2$, where the first element represents the maximum row feature of the corresponding matrix, and the second element represents the maximum column feature of the corresponding matrix. Therefore, the resulting statistical vector preserves important information and includes positional details.

Due to the nature of the ReLU activation function, all elements in the feature maps are non-negative. Therefore, the absolute value operation in the L₁ norm becomes redundant, i.e., $|u_{ij}| = u_{ij}$. This simplification eliminates the need for absolute value calculations, although the $\max$ operation remains sub-differentiable. In practice, subgradients can be used during backpropagation, making the overall pooling operation effectively differentiable. We can simplify the operation as:

$$\left[\begin{array}{c} \max_j \sum_i |u_{ij}| \\ \max_i \sum_j |u_{ij}| \end{array}\right] = \left[\begin{array}{c} \max_j \sum_i u_{ij} \\ \max_i \sum_j u_{ij} \end{array}\right]$$

- $z_1$ : **maximum column sum** – captures the strongest vertical strip (induced $L_1$ norm)
- $z_2$ : **maximum row sum** – captures the strongest horizontal strip (induced $L_\infty$ norm)

These two values form a compact 2-dimensional descriptor that retains the most energetic rows and columns, preserving both **intensity** and **spatial sensitivity**.

The figure below illustrates the HV-Pooling operation:

<div align="center">
  <img src="hv_pooling.png" alt="HV-Pooling mechanism" width="400"/>
  <p><em>Our horizontal and vertical pooling method operates on a feature map denoted as <i>M</i>. The functions <i>f<sub>1</sub></i> and <i>f<sub>2</sub></i> serve as mappings that reduce the matrix to statistical quantities. Specifically, <i>f<sub>1</sub></i> extracts the maximum feature from each column, while <i>f<sub>2</sub></i> extracts the maximum feature from each row.</em></p>
</div>

---

## 🛠️ Environment & Requirements

Experiments were conducted on the following setup:

| Component | Specification |
|-----------|---------------|
| **OS** | Ubuntu 18.04 |
| **GPU** | NVIDIA GeForce RTX 3090 (24GB VRAM) |
| **CUDA** | 11.0 (compatible with PyTorch 1.7) |
| **Python** | 3.8 |
| **PyTorch** | 1.7 |

---

# HV-Pooling: Theoretical Properties

Let *U* ∈ ℝ<sup>H×W</sup> denote a non-negative feature map output by a convolutional layer followed by ReLU activation. Define:

- **Maximum element**: *m* = max<sub>i,j</sub> u<sub>ij</sub>
- **Global Average Pooling (GAP)**: *g* = (1/(H×W)) ∑<sub>i=1</sub><sup>H</sup> ∑<sub>j=1</sub><sup>W</sup> u<sub>ij</sub>
- **Maximum column sum** (induced L<sub>1</sub> norm): *c* = ‖U‖<sub>1</sub> = max<sub>1≤j≤W</sub> ∑<sub>i=1</sub><sup>H</sup> u<sub>ij</sub>
- **Maximum row sum** (induced L<sub>∞</sub> norm): *r* = ‖U‖<sub>∞</sub> = max<sub>1≤i≤H</sub> ∑<sub>j=1</sub><sup>W</sup> u<sub>ij</sub>
- **HV-Pooling output vector**: **z** = [c, r]<sup>T</sup>, and its combined value: *h* = (c + r)/2

---

## Theorem 1: Intensity Preservation

For any non-negative feature map *U*, we have

*c* ≥ *m*, *r* ≥ *m*,

and consequently *h* ≥ *m*. In contrast, the GAP output satisfies *g* ≤ *m*, and when the feature map is sparse, *g* can be much smaller than *m*.

### Proof

Let (i*, j*) be the position of the maximum element, i.e., u<sub>i*j*</sub> = *m*. Consider the sum of the j*-th column:

∑<sub>i=1</sub><sup>H</sup> u<sub>i j*</sub> ≥ u<sub>i*j*</sub> = *m*,

so this column sum is at least *m*, hence *c* = max<sub>j</sub> ∑<sub>i</sub> u<sub>ij</sub> ≥ *m*. Similarly, the sum of the i*-th row is at least *m*, so *r* ≥ *m*. Therefore *h* ≥ *m*.

On the other hand, by the inequality of arithmetic means,

*g* = (1/(H×W)) ∑<sub>i,j</sub> u<sub>ij</sub> ≤ (1/(H×W)) ∑<sub>i,j</sub> *m* = *m*,

with equality if and only if all elements are equal. When the feature map is sparse (i.e., most elements are zero), let *k* be the number of non-zero elements, and denote the proportion of non-zero elements as ρ = k/(H×W) ≪ 1. If all non-zero elements equal *m*, then

*g* = (k × m)/(H×W) = ρ × m ≪ m,

while the HV-Pooling output still satisfies *h* ≥ *m*. Thus, **HV-Pooling preserves the intensity of the maximum element**, whereas GAP dilutes it. ∎

---

## Theorem 2: Spatial Location Sensitivity

GAP is completely insensitive to spatial location transformations of the feature map, i.e., for any position transformation (element rearrangement) π, we have *g*(π(*U*)) = *g*(*U*). In contrast, **HV-Pooling is sensitive to spatial location transformations**, meaning there exist transformations that change its output values. Therefore, it can reflect the spatial structure information of the feature map.

### Proof

The mathematical definition of GAP is:

*g*(*U*) = (1/(H×W)) ∑<sub>i=1</sub><sup>H</sup> ∑<sub>j=1</sub><sup>W</sup> u<sub>ij</sub>

Since any position transformation π only changes the positions of elements without altering their values, the sum of all elements remains unchanged. Thus, *g*(π(*U*)) = *g*(*U*) holds for all π.

For HV-Pooling, the maximum column sum and maximum row sum are defined as:

*c*(*U*) = max<sub>j</sub> ∑<sub>i</sub> u<sub>ij</sub>, *r*(*U*) = max<sub>i</sub> ∑<sub>j</sub> u<sub>ij</sub>

A position transformation π reassigns the elements of the original feature map to new rows and columns. After transformation, the row sums and column sums of the new feature map are composed of certain combinations of the original elements. Since the spatial distribution of elements changes, the row sum vector and column sum vector generally change accordingly.

Consider a transformation π that:
- Concentrates important elements originally scattered across different rows into the same row, or
- Disperses elements originally concentrated in one column across different columns

Such rearrangements alter the values of row sums and column sums, potentially causing max<sub>i</sub> ∑<sub>j</sub> u<sub>ij</sub> or max<sub>j</sub> ∑<sub>i</sub> u<sub>ij</sub> to change. Therefore, there exist position transformations such that *c*(π(*U*)) ≠ *c*(*U*) or *r*(π(*U*)) ≠ *r*(*U*).

This demonstrates that **the output of HV-Pooling depends on the spatial arrangement of elements**, is sensitive to position transformations, and can reflect the spatial structure information of the feature map. ∎

---

## Summary Table

| Property | GAP | HV-Pooling |
|----------|-----|------------|
| Intensity Preservation | ❌ Dilutes max value | ✅ Preserves max value |
| Spatial Sensitivity | ❌ Invariant to rearrangement | ✅ Sensitive to structure |
