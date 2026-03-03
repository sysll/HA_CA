import torch
import torch.nn as nn
import torch.nn.functional as F


class HVCA(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(2 * channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_abs = x.abs()
        v = x_abs.sum(dim=2).amax(dim=2)
        h_ = x_abs.sum(dim=3).amax(dim=2)
        z = torch.cat([v, h_], dim=1)
        a = self.fc2(self.act(self.fc1(z)))
        a = self.gate(a).view(b, c, 1, 1)
        return x * a


def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    x = x.view(-1, window_size * window_size, C)
    return x


def window_unpartition(x, window_size, B, C, H, W):
    nH = H // window_size
    nW = W // window_size
    x = x.view(B, nH, nW, window_size, window_size, C)
    x = x.permute(0, 5, 3, 1, 4, 2).contiguous()
    x = x.view(B, C, H, W)
    return x


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=8, mlp_ratio=4.0):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        xw = window_partition(x, self.window_size)
        xw = self.norm1(xw)
        attn_out, _ = self.attn(xw, xw, xw)
        xw = attn_out + xw
        xw = self.norm2(xw)
        xw = self.mlp(xw) + xw
        x = window_unpartition(xw, self.window_size, B, C, H, W)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=4, stride=4)

    def forward(self, x):
        return self.proj(x)


class SwinStage(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=8):
        super().__init__()
        self.blocks = nn.Sequential(*[
            SwinBlock(dim, num_heads=num_heads, window_size=window_size)
            for _ in range(depth)
        ])
        self.down = nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.blocks(x)
        x = self.down(skip)
        return skip, x


class SwinUNetWithHVCA(nn.Module):
    def __init__(self, in_ch=3, num_classes=4, base_dim=96, reduction=16):
        super().__init__()

        self.patch_embed = PatchEmbed(in_ch=in_ch, embed_dim=base_dim)

        self.stage1 = SwinStage(base_dim,     depth=2, num_heads=3,  window_size=8)
        self.stage2 = SwinStage(base_dim*2,   depth=2, num_heads=6,  window_size=8)
        self.stage3 = SwinStage(base_dim*4,   depth=2, num_heads=12, window_size=8)
        self.stage4 = SwinStage(base_dim*8,   depth=2, num_heads=24, window_size=8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_dim*16, base_dim*16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_dim*16, base_dim*16, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_dim*16, base_dim*8, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(base_dim*16, base_dim*8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_dim*8,  base_dim*8, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.hvca4 = HVCA(base_dim*8, reduction=reduction)

        self.up3 = nn.ConvTranspose2d(base_dim*8, base_dim*4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_dim*8, base_dim*4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_dim*4, base_dim*4, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.hvca3 = HVCA(base_dim*4, reduction=reduction)

        self.up2 = nn.ConvTranspose2d(base_dim*4, base_dim*2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_dim*4, base_dim*2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_dim*2, base_dim*2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.hvca2 = HVCA(base_dim*2, reduction=reduction)

        self.up1 = nn.ConvTranspose2d(base_dim*2, base_dim, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_dim*2, base_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_dim,   base_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.hvca1 = HVCA(base_dim, reduction=reduction)

        self.final_conv = nn.Conv2d(base_dim, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.patch_embed(x)

        skip1, x = self.stage1(x)
        skip2, x = self.stage2(x)
        skip3, x = self.stage3(x)
        skip4, x = self.stage4(x)

        x = self.bottleneck(x)

        d = self.up4(x)
        d = torch.cat([d, skip4], dim=1)
        d = self.dec4(d)
        d = self.hvca4(d)

        d = self.up3(d)
        d = torch.cat([d, skip3], dim=1)
        d = self.dec3(d)
        d = self.hvca3(d)

        d = self.up2(d)
        d = torch.cat([d, skip2], dim=1)
        d = self.dec2(d)
        d = self.hvca2(d)

        d = self.up1(d)
        d = torch.cat([d, skip1], dim=1)
        d = self.dec1(d)
        d = self.hvca1(d)

        out = self.final_conv(d)
        return out


if __name__ == "__main__":
    model = SwinUNetWithHVCA(
        in_ch=3,
        num_classes=4,
        base_dim=96,
        reduction=16
    ).cuda()

    dummy = torch.randn(2, 3, 256, 256).cuda()
    out = model(dummy)
    print(out.shape)  # should be [2, 4, 64, 64]