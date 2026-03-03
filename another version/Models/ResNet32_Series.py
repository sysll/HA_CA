import torch
import torch.nn as nn
import math

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# 残差网络中的basicblock(在resnet18和resnet34中用到)
class BasicBlock(nn.Module):
    expansion = 1  # 在每个basicblock中有的两个conv大小通道数都一样，所以这边expansion = 1

    # inplanes代表输入通道数，planes代表输出通道数
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 这边下采样的作用是，我们知道resnet一共有5个stage（0，1，2，3，4），第1-4个都是由block组成的，但stage1的第一个block不进行下采样
        # 也就是说它的stride=1，所以除了stage1的shortcut不需要进行下采样以外，其他的都需要；因为2-4的第一个block，stride=2
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



"""一般的ResNet32"""
class ResNet32(nn.Module):
    def __init__(self, block, layers, num_classes=1000):  # layers=参数列表
        #对于stage0，大家都是一样的
        self.inplanes = 64
        super(ResNet32, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #从stage1开始不一样
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

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
        x = x.view(x.size(0), -1)   # 将输出结果展成一行
        x = self.fc(x)

        return x


def Get_ResNet32(num_classes):
    return ResNet32(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)






"""SE-ResNet"""
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SE_ResNet32(nn.Module):
    def __init__(self, block, layers, num_classes=2):  # layers=参数列表
        #对于stage0，大家都是一样的
        self.inplanes = 64
        super(SE_ResNet32, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #从stage1开始不一样
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.channel_attention = ChannelAttention(64)
        self.channel_attention1 = ChannelAttention(64)
        self.channel_attention2 = ChannelAttention(128)
        self.channel_attention3 = ChannelAttention(256)
        self.channel_attention4 = ChannelAttention(512)


        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.channel_attention(x)

        x = self.layer1(x)
        x = self.channel_attention1(x)

        x = self.layer2(x)
        x = self.channel_attention2(x)

        x = self.layer3(x)
        x = self.channel_attention3(x)

        x = self.layer4(x)
        x = self.channel_attention4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)   # 将输出结果展成一行
        x = self.fc(x)

        return x



def Get_SE_ResNet32(num_classes):
    return SE_ResNet32(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)







"""LVPN-ResNet18"""
def Level_vertical_pooling(x):
    L_inf = torch.max(torch.sum(torch.abs(x), dim=3), dim=2).values.unsqueeze(2)
    L1 = torch.max(torch.sum(torch.abs(x), dim=2), dim=2).values.unsqueeze(2)
    feature_cat_vec = torch.cat((L_inf, L1), dim=2).flatten(1)
    return feature_cat_vec

class LVP_ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(LVP_ChannelAttention, self).__init__()
        self.LVP = Level_vertical_pooling
        self.fc1 = nn.Linear(2 * in_planes, in_planes // 4)
        self.relu1 = nn.Mish()
        self.fc2 = nn.Linear( in_planes // 4, in_planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        tmp = x
        x = self.LVP(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x).unsqueeze(2).unsqueeze(3)
        x = self.sigmoid(x) * tmp
        return x
class LVPN_ResNet32(nn.Module):
    def __init__(self, block, layers, num_classes=2):  # layers=参数列表
        #对于stage0，大家都是一样的
        self.inplanes = 64
        super(LVPN_ResNet32, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #从stage1开始不一样
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.channel_attention = LVP_ChannelAttention(64)
        self.channel_attention1 = LVP_ChannelAttention(64)
        self.channel_attention2 = LVP_ChannelAttention(128)
        self.channel_attention3 = LVP_ChannelAttention(256)
        self.channel_attention4 = LVP_ChannelAttention(512)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.channel_attention(x)

        x = self.layer1(x)
        x = self.channel_attention1(x)

        x = self.layer2(x)
        x = self.channel_attention2(x)

        x = self.layer3(x)
        x = self.channel_attention3(x)

        x = self.layer4(x)
        x = self.channel_attention4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 将输出结果展成一行
        x = self.fc(x)

        return x


def Get_LVPN_ResNet32(num_classes):
    return LVPN_ResNet32(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)