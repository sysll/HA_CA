import argparse
from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from models.unet_hvca import UNetHVCA
from models.swin_unet import SwinUnet

class CardiacDataset(Dataset):
    def __init__(self, img_dir: Path, mask_dir: Path, img_size: int):
        try:
            from PIL import Image
        except Exception:
            raise RuntimeError("Pillow is required for segmentation dataset loader")
        self.Image = Image
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.items = sorted([p for p in img_dir.glob("*") if p.is_file()])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p = self.items[idx]
        img = self.Image.open(p).convert("RGB").resize((self.img_size, self.img_size))
        m = self.Image.open(self.mask_dir / p.name).convert("L").resize((self.img_size, self.img_size))
        x = torch.frombuffer(img.tobytes(), dtype=torch.uint8).float().view(self.img_size, self.img_size, 3).permute(2,0,1) / 255.0
        y = torch.frombuffer(m.tobytes(), dtype=torch.uint8).long().view(self.img_size, self.img_size)
        return x, y

def dice_loss(logits, targets, num_classes: int):
    probs = torch.sigmoid(logits) if num_classes == 1 else torch.softmax(logits, dim=1)
    if num_classes == 1:
        targets_onehot = targets.float()
        inter = (probs[:,0] * targets_onehot).sum()
        union = probs[:,0].sum() + targets_onehot.sum()
        return 1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)
    else:
        bs, c, h, w = probs.shape
        targets_onehot = torch.zeros(bs, c, h, w, device=probs.device)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1.0)
        inter = (probs * targets_onehot).sum(dim=(0,2,3)).mean()
        union = probs.sum(dim=(0,2,3)).mean() + targets_onehot.sum(dim=(0,2,3)).mean()
        return 1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)

def seg_metrics(logits, targets, num_classes: int):
    if num_classes == 1:
        probs = torch.sigmoid(logits)
        pred = (probs[:,0] > 0.5).long()
    else:
        pred = logits.argmax(dim=1)
    dice_list = []
    iou_list = []
    sens_list = []
    prec_list = []
    for c in range(num_classes if num_classes > 1 else 2):
        if num_classes == 1:
            t = (targets == 1).long()
            p = pred
        else:
            t = (targets == c).long()
            p = (pred == c).long()
        tp = (p * t).sum().float()
        fp = (p * (1 - t)).sum().float()
        fn = ((1 - p) * t).sum().float()
        denom_dice = (2 * tp + fp + fn + 1e-6)
        dice = (2 * tp + 1e-6) / denom_dice
        denom_iou = (tp + fp + fn + 1e-6)
        iou = (tp + 1e-6) / denom_iou
        sens = (tp + 1e-6) / (tp + fn + 1e-6)
        prec = (tp + 1e-6) / (tp + fp + 1e-6)
        dice_list.append(dice.item())
        iou_list.append(iou.item())
        sens_list.append(sens.item())
        prec_list.append(prec.item())
    return sum(dice_list)/len(dice_list), sum(iou_list)/len(iou_list), sum(sens_list)/len(sens_list), sum(prec_list)/len(prec_list)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=str, required=True)
    p.add_argument("--masks", type=str, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--reduction", type=int, default=16)
    p.add_argument("--model", type=str, default="swinunet")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CardiacDataset(Path(args.images), Path(args.masks), args.img_size)
    idxs = list(range(len(ds)))
    random.Random(42).shuffle(idxs)
    split = int(0.8 * len(ds))
    train_idx = idxs[:split]
    val_idx = idxs[split:]
    train_dl = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=0)
    if args.model == "swinunet":
        model = SwinUnet(in_ch=3, num_classes=args.num_classes).to(device)
    elif args.model == "unet_hvca":
        model = UNetHVCA(in_ch=3, num_classes=args.num_classes, base=64, reduction=args.reduction).to(device)
    else:
        raise ValueError(args.model)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss() if args.num_classes == 1 else nn.CrossEntropyLoss()
    for e in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y if args.num_classes > 1 else y.float().unsqueeze(1))
            loss.backward()
            opt.step()
            total += float(loss)
        model.eval()
        with torch.no_grad():
            m_dice = 0.0
            m_iou = 0.0
            m_sens = 0.0
            m_prec = 0.0
            for x, y in val_dl:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                d, i, s, p = seg_metrics(logits, y, args.num_classes)
                m_dice += d
                m_iou += i
                m_sens += s
                m_prec += p
            n = max(1, len(val_dl))
            print(e, total / max(1, len(train_dl)), m_dice / n, m_iou / n, m_sens / n, m_prec / n)

if __name__ == "__main__":
    main()
