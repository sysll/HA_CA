import argparse
from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

def build_model(name: str, num_classes: int, reduction: int, attn: str):
    from models.wrappers import VGGWithAttn, MobileNetWithAttn, DenseNetWithAttn, EfficientNetWithAttn
    from models.resnet_hvca import ResNetHVCA
    import torchvision.models as tvm
    if name == "resnet18":
        backbone = tvm.resnet18(num_classes=num_classes)
        return ResNetHVCA(backbone, reduction=reduction) if attn == "hvca" else backbone
    if name == "resnet34":
        backbone = tvm.resnet34(num_classes=num_classes)
        return ResNetHVCA(backbone, reduction=reduction) if attn == "hvca" else backbone
    if name == "resnet50":
        backbone = tvm.resnet50(num_classes=num_classes)
        return ResNetHVCA(backbone, reduction=reduction) if attn == "hvca" else backbone
    if name == "vgg16":
        backbone = tvm.vgg16(num_classes=num_classes)
        return VGGWithAttn(backbone, attn_type=attn, reduction=reduction)
    if name == "mobilenet_v2":
        backbone = tvm.mobilenet_v2(num_classes=num_classes)
        return MobileNetWithAttn(backbone, attn_type=attn, reduction=reduction)
    if name == "densenet121":
        backbone = tvm.densenet121(num_classes=num_classes)
        return DenseNetWithAttn(backbone, attn_type=attn, reduction=reduction)
    if name == "efficientnet_b0":
        backbone = tvm.efficientnet_b0(num_classes=num_classes)
        return EfficientNetWithAttn(backbone, attn_type=attn, reduction=reduction)
    if name == "vit_b_16":
        backbone = tvm.vit_b_16(num_classes=num_classes)
        return backbone
    raise ValueError(name)

def build_dataset(root: Path, img_size: int):
    import torchvision.transforms as T
    import torchvision.datasets as D
    tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    return D.ImageFolder(str(root), transform=tf)

def split_kfold(n: int, k: int, seed: int):
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    folds = []
    fold_size = n // k
    for i in range(k):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else n
        val_idx = idxs[val_start:val_end]
        train_idx = idxs[:val_start] + idxs[val_end:]
        folds.append((train_idx, val_idx))
    return folds

def accuracy(logits, targets):
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()

def precision_recall_f1(logits, targets, num_classes: int):
    pred = logits.argmax(dim=1)
    tp = torch.zeros(num_classes, device=targets.device)
    fp = torch.zeros(num_classes, device=targets.device)
    fn = torch.zeros(num_classes, device=targets.device)
    for c in range(num_classes):
        tp[c] = ((pred == c) & (targets == c)).sum()
        fp[c] = ((pred == c) & (targets != c)).sum()
        fn[c] = ((pred != c) & (targets == c)).sum()
    prec = (tp / (tp + fp + 1e-8)).mean().item()
    rec = (tp / (tp + fn + 1e-8)).mean().item()
    f1 = (2 * prec * rec / (prec + rec + 1e-8)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def train_epoch(model, dl, opt, loss_fn, device):
    model.train()
    total = 0.0
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        total += float(loss)
    return total / max(1, len(dl))

def eval_epoch(model, dl, loss_fn, device, num_classes: int):
    model.eval()
    total = 0.0
    acc_sum = 0.0
    p_sum = 0.0
    r_sum = 0.0
    f_sum = 0.0
    cm = torch.zeros(num_classes, num_classes, device=device)
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total += float(loss)
            acc_sum += accuracy(logits, y)
            p, r, f1 = precision_recall_f1(logits, y, num_classes)
            p_sum += p
            r_sum += r
            f_sum += f1
            pred = logits.argmax(dim=1)
            for i in range(num_classes):
                for j in range(num_classes):
                    cm[i, j] += ((y == i) & (pred == j)).sum()
    n = max(1, len(dl))
    return total / n, acc_sum / n, p_sum / n, r_sum / n, f_sum / n, cm.long().cpu().tolist()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--model", type=str, default="resnet18")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--reduction", type=int, default=16)
    p.add_argument("--attn", type=str, default="hvca")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.dataset_root)
    ds = build_dataset(root, args.img_size)
    num_classes = len(ds.classes)
    folds = split_kfold(len(ds.samples), args.folds, args.seed)
    loss_fn = nn.CrossEntropyLoss()
    for i, (train_idx, val_idx) in enumerate(folds):
        print(f"fold {i+1}/{args.folds}")
        train_dl = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_dl = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)
        model = build_model(args.model, num_classes=num_classes, reduction=args.reduction, attn=args.attn).to(device)
        opt = optim.Adam(model.parameters(), lr=args.lr)
        for e in range(args.epochs):
            tr_loss = train_epoch(model, train_dl, opt, loss_fn, device)
            vl_loss, vl_acc, vl_p, vl_r, vl_f1, vl_cm = eval_epoch(model, val_dl, loss_fn, device, num_classes)
            print(e, tr_loss, vl_loss, vl_acc, vl_p, vl_r, vl_f1, vl_cm)

if __name__ == "__main__":
    main()
