# HV-CA 实验与训练入口

本仓库提供论文实验的完整训练脚本与统一接口，包括两类医学数据集（五折训练）与三类自然图像数据集。

## 数据目录
- `data/skin_cancer/类名/图片`（ImageFolder）
- `data/sakha_tb/类名/图片`（ImageFolder）
- `data/cardiacuda/images/*.png|jpg`
- `data/cardiacuda/masks/*.png`（与 images 同名）
- `data/cifar100/类名/图片`（ImageFolder）
- `data/imagenet100/类名/图片`（ImageFolder）

## 模型与注意力接口
- 分类主干：`resnet18|resnet34|resnet50|vgg16|mobilenet_v2|densenet121|efficientnet_b0|vit_b_16`
- 注意力选项：`none|hvca|se|eca|ca|fca|ortho|epsa|slcam`
- 压缩比：`--reduction`（默认 16）

## 医学数据五折训练
- 皮肤癌（ResNet18+HVCA）
  - `.\.venv\Scripts\python.exe experiments\train_skin_cancer_kfold.py`
  - 评估：每折输出 loss、ACC、Precision、Recall、F1、Confusion Matrix
- Sakha-TB（ResNet34+HVCA）
  - `.\.venv\Scripts\python.exe experiments\train_sakha_tb_kfold.py`
  - 评估同上

如需自定义模型或注意力：
```
.\.venv\Scripts\python.exe trainers\train_medical_kfold.py ^
  --dataset_root data\skin_cancer ^
  --model resnet50 ^
  --attn fca ^
  --epochs 30 --batch_size 64 --lr 1e-2 --img_size 224 --reduction 16 --folds 5 --seed 42
```

## 分割（CardiacUDA，主模型 SwinUnet）
- `.\.venv\Scripts\python.exe experiments\train_cardiacuda.py`
- 训练/验证划分：固定 8:2
- 指标：mean Dice、mean IoU、Sensitivity、Precision
- 可选模型：`--model swinunet|unet_hvca`

## 自然图像数据
- CIFAR-100（ResNet50+HVCA，100 epoch）
  - `.\.venv\Scripts\python.exe experiments\train_cifar100.py`
- ImageNet-100（ResNet50+HVCA，100 epoch）
  - `.\.venv\Scripts\python.exe experiments\train_imagenet100.py`

如需切换注意力：
```
.\.venv\Scripts\python.exe trainers\train_cls.py ^
  --dataset_root data\cifar100 ^
  --model resnet50 ^
  --attn epsa ^
  --epochs 100 --batch_size 128 --lr 1e-2 --img_size 224 --reduction 16
```

## 依赖
- Python 3.8+
- PyTorch
- Torchvision（分类骨干与数据）
- Pillow（分割数据加载）

## 指标说明
- 分类：ACC、宏平均 Precision/Recall/F1、混淆矩阵
- 分割：mean Dice、mean IoU、Sensitivity、Precision

## 备注
- HV-CA 模块位于 `models/hvca.py`
- 其他注意力模块位于 `models/attn/`，统一通过 `models/wrappers.py` 接入分类主干
- 医学 5 折训练通用管线：`trainers/train_medical_kfold.py`
