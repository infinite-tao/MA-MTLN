"""

For fair comparison, we employed SENet-50 as the backbone of the network.

"""

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.LeakyReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid())
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

class SENetBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None, reduction=16):
        super(SENetBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(32, mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.gn2 = nn.GroupNorm(32, mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.gn3 = nn.GroupNorm(32, planes * self.expansion)
        self.relu = nn.PReLU()
        self.downsample = downsample
        self.se = SELayer3D(planes * self.expansion, reduction)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.gn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SENetDilatedBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(SENetDilatedBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(32, mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=2,
            dilation=2,
            groups=cardinality,
            bias=False)
        self.gn2 = nn.GroupNorm(32, mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.gn3 = nn.GroupNorm(32, planes * self.expansion)
        self.relu = nn.PReLU()
        self.downsample = downsample
        self.se = SELayer3D(planes * self.expansion, reduction=16)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.gn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class set_conv(nn.Module):
    def __init__(self, nchannel):
        super(set_conv, self).__init__()
        self.conv = nn.Conv3d(nchannel, nchannel, kernel_size=5, padding=2)
        self.bn = nn.GroupNorm(32, nchannel)
        self.relu = nn.PReLU()
    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out

def chioce_set(nchannel, depth):
        layer = []
        for _ in range(depth):
            layer.append(set_conv(nchannel))
        return nn.Sequential(*layer)

class Uptranspose(nn.Module):
    def __init__(self, inchannel, outchannel, nconv, stride, pad, outpad):
        super(Uptranspose,self).__init__()
        self.up_conv = nn.ConvTranspose3d(inchannel, outchannel//2, kernel_size=3, stride=stride, padding=pad, output_padding=outpad)
        self.gn = nn.GroupNorm(32, outchannel//2)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.ops = chioce_set(outchannel, nconv)

    def forward(self, x, skipx):
        out = self.relu1(self.gn(self.up_conv(x)))
        xcat = torch.cat((out, skipx), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out

class CmsVNet(nn.Module):

    def __init__(self, block, layers, shortcut_type='B', cardinality=32, num_classes=2):
        self.inplanes = 32
        super(CmsVNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.gn = nn.GroupNorm(16, 32)
        self.relu = nn.PReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0], shortcut_type, cardinality)
        self.layer2 = self._make_layer(block, 64, layers[1], shortcut_type, cardinality, stride=(1, 2, 2))
        self.layer3 = self._make_layer(block, 128, layers[2], shortcut_type, cardinality, stride=(1, 2, 2))
        self.layer4 = self._make_layer(SENetDilatedBottleneck, 256, layers[3], shortcut_type, cardinality, stride=1)
        # self.layer4 = self._make_layer(SENetDilatedBottleneck, 512, layers[3], shortcut_type, cardinality, stride=1)
        self.up_layer5 = Uptranspose(512, 512, 2, 1, 1, 0)
        self.up_layer6 = Uptranspose(512, 256, 2, (1, 2, 2), 1, (0, 1, 1))
        self.up_layer7 = Uptranspose(256, 128, 1, (1, 2, 2), 1, (0, 1, 1))
        self.up_layer8 = Uptranspose(128, 64, 1, 1, 1, 0)
        self.out = nn.Conv3d(64, 2, kernel_size=1)
        self.GAP1 = nn.AdaptiveAvgPool3d(1)
        self.GAP2 = nn.AdaptiveAvgPool3d(1)
        self.GAP3 = nn.AdaptiveAvgPool3d(1)
        self.FC1 = nn.Linear(1280, 128)
        self.FC2 = nn.Linear(128, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False),
                    nn.GroupNorm(32, planes * block.expansion)
                )

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):

        x0 = self.conv1(x)
        x0 = self.gn(x0)
        x0 = self.relu(x0)
        out32 = self.maxpool(x0)
        out64 = self.layer1(out32)
        out128 = self.layer2(out64)
        out256 = self.layer3(out128)

        out_mid = self.layer4(out256)

        out512 = self.up_layer5(out_mid, out256)
        out = self.up_layer6(out512, out128)
        out = self.up_layer7(out, out64)
        out = self.up_layer8(out, out32)
        out = F.upsample(out, size=x.size()[2:], mode='trilinear')
        out = self.out(out)

        x00 = self.GAP1(out256)
        x10 = self.GAP2(out_mid)
        x20 = self.GAP3(out512)
        x3 = torch.cat((x00, x10, x20), 1)
        x4 = x3.view(x3.size(0), -1)
        x5 = self.FC2(self.FC1(x4))
        return out, x5


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters

def CmsVNet50(**kwargs):
    """Constructs a SENet-3D-50 model."""
    model = CmsVNet(SENetBottleneck, [3, 4, 6, 3], **kwargs)
    return model