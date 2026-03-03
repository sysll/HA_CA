import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

eps = 1.0e-5


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


# ======================================
#          Attention Modules
# ======================================

class CALayer(nn.Module):
    """Coordinate Attention"""
    def __init__(self, inp, groups=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // groups)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.relu = nn.Hardswish()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)
        y = identity * x_w * x_h
        return y


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc2(self.fc1(self.pool(x))))


class ChannelAttention_avmax(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.cv1(torch.cat([
            torch.mean(x, 1, keepdim=True),
            torch.max(x, 1, keepdim=True)[0]
        ], 1)))


class CBAMLayer(nn.Module):
    def __init__(self, c1, kernel_size=7, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(c1, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))


class SLCAMLayer(nn.Module):
    """Spatial + Location + Channel Attention Module (your custom one)"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.cv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.c1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)
        self.c2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.x_ca = ChannelAttention_avmax(channel, reduction)

    def forward(self, x):
        _, _, h, w = x.size()
        # spatial
        x_c = self.sigmoid(self.cv1(torch.cat([
            torch.mean(x, 1, keepdim=True),
            torch.max(x, 1, keepdim=True)[0]
        ], 1)))
        # height
        x_h = self.sigmoid(
            self.c2(self.relu(self.bn(self.c1(
                torch.mean(x, 2, keepdim=True) + torch.max(x, 2, keepdim=True)[0]
            ))))
        )
        # width
        x_w = self.sigmoid(
            self.c2(self.relu(self.bn(self.c1(
                torch.mean(x, 3, keepdim=True) + torch.max(x, 3, keepdim=True)[0]
            ))))
        )
        out = x * x_h * x_w * x_c
        return self.x_ca(out)


# ======================================
#          Basic Blocks
# ======================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dilation=1,
                 downsample=None, with_cp=False, drop_path_rate=0.0,
                 CA=False, CBAM=False, SLCAM=False):
        super().__init__()
        self.with_cp = with_cp
        self.downsample = downsample
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > eps else nn.Identity()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.att = None
        if CBAM:
            self.att = CBAMLayer(out_channels)
        if CA:
            self.att = CALayer(out_channels)
        if SLCAM:
            self.att = SLCAMLayer(out_channels)

    def forward(self, x):
        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.att is not None:
                out = self.att(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out = self.drop_path(out) + identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, dilation=1,
                 downsample=None, style='pytorch', with_cp=False,
                 drop_path_rate=0.0, CA=False, CBAM=False, SLCAM=False):
        super().__init__()
        self.with_cp = with_cp
        self.downsample = downsample
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > eps else nn.Identity()
        self.style = style

        mid_channels = out_channels // self.expansion

        if style == 'pytorch':
            conv1_stride = 1
            conv2_stride = stride
        else:
            conv1_stride = stride
            conv2_stride = 1

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, stride=conv1_stride, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=conv2_stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.att = None
        if CBAM:
            self.att = CBAMLayer(out_channels)
        if CA:
            self.att = CALayer(out_channels)
        if SLCAM:
            self.att = SLCAMLayer(out_channels)

    def forward(self, x):
        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv3(out)
            out = self.bn3(out)
            if self.att is not None:
                out = self.att(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out = self.drop_path(out) + identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)
        return out


# ======================================
#          ResNet Backbone
# ======================================

class ResNet(nn.Module):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth=50,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(3,),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 with_cp=False,
                 zero_init_residual=True,
                 CA=False,
                 CBAM=False,
                 SLCAM=False,
                 drop_path_rate=0.0):
        super().__init__()

        if depth not in self.arch_settings:
            raise ValueError(f"depth {depth} not in supported {list(self.arch_settings.keys())}")

        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.strides = strides
        self.dilations = dilations
        self.out_indices = out_indices
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = nn.ModuleList()
        in_channels = stem_channels
        out_channels = base_channels * self.block.expansion

        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]

            downsample = None
            if stride != 1 or in_channels != out_channels:
                conv_stride = stride
                if self.avg_down and stride != 1:
                    conv_stride = 1
                    downsample = nn.Sequential(
                        nn.AvgPool2d(kernel_size=stride, stride=stride,
                                     ceil_mode=True, count_include_pad=False),
                        nn.Conv2d(in_channels, out_channels, 1, stride=conv_stride, bias=False),
                        nn.BatchNorm2d(out_channels),
                    )
                else:
                    downsample = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, stride=conv_stride, bias=False),
                        nn.BatchNorm2d(out_channels),
                    )

            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=num_blocks,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                style=self.style,
                with_cp=self.with_cp,
                drop_path_rate=drop_path_rate,
                CA=CA,
                CBAM=CBAM,
                SLCAM=SLCAM
            )
            self.res_layers.append(res_layer)

            in_channels = out_channels
            out_channels *= 2

        self._freeze_stages()

        self.feat_dim = in_channels  # last stage output channels

    def make_res_layer(self, block, num_blocks, in_channels, out_channels,
                       stride=1, dilation=1, downsample=None, **kwargs):
        layers = []
        layers.append(block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            **kwargs
        ))
        for _ in range(1, num_blocks):
            layers.append(block(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                dilation=dilation,
                **kwargs
            ))
        return nn.Sequential(*layers)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, stem_channels // 2, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_channels // 2, stem_channels // 2, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(stem_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_channels // 2, stem_channels, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, stem_channels, 7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(stem_channels)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                for p in self.stem.parameters():
                    p.requires_grad = False
            else:
                for p in [self.conv1, self.bn1]:
                    for param in p.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.res_layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for i, layer in enumerate(self.res_layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()


# ======================================
#          Variants (optional)
# ======================================

class ResNetV1c(ResNet):
    def __init__(self, **kwargs):
        super().__init__(deep_stem=True, avg_down=False, **kwargs)


class ResNetV1d(ResNet):
    def __init__(self, **kwargs):
        super().__init__(deep_stem=True, avg_down=True, **kwargs)


# ======================================
#          Test
# ======================================

if __name__ == '__main__':
    import sys
    import io

    # 解决 Windows cmd 下 emoji 乱码问题
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    x = torch.rand(2, 3, 224, 224)
    model = ResNet(50, SLCAM=True)
    outs = model(x)

    print("Output shapes:")
    for out in outs:
        print(out.shape)

    print("纯 PyTorch ResNet + SLCAM 运行成功！")