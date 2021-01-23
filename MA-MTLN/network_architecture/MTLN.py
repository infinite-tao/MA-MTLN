from torch import nn
import torch
import torch.nn.functional as F
from network_architecture.BackBone import BackBone3D
from network_architecture.BiVAblock import BiVA
# from neural_network import SegmentationNetwork


def add_conv3D(in_ch, out_ch, ksize, stride):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv3d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('group_norm', nn.GroupNorm(16, out_ch))
    stage.add_module('leaky', nn.PReLU())
    return stage

class ASA3D(nn.Module):
    def __init__(self, level, vis=False):
        super(ASA3D, self).__init__()
        self.level = level
        compress_c = 16
        self.attention = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.pool = add_conv3D(64, 64, 3, (1, 2, 2))
        self.weight_level_0 = add_conv3D(64, compress_c, 1, 1)
        self.weight_level_1 = add_conv3D(64, compress_c, 1, 1)

        self.weight_levels = nn.Conv3d(compress_c*2, 2, kernel_size=1, stride=1, padding=0)
        self.vis = vis


    def forward(self, inputs0, inputs1):
        if self.level == 0:
            level_f = inputs1
        elif self.level == 1:
            level_f = self.pool(inputs1)
        elif self.level == 2:
            level_f0 = self.pool(inputs1)
            level_f = self.pool(level_f0)
        elif self.level == 3:
            level_f0 = self.pool(inputs1)
            level_f = self.pool(level_f0)

        level_0_weight_v = self.weight_level_0(inputs0)
        level_1_weight_v = self.weight_level_1(level_f)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        adaptive_attention = self.attention(inputs0) * inputs0 * levels_weight[:, 0:1, :, :, :] + \
                            self.attention(level_f) * level_f * levels_weight[:, 1:, :, :, :]

        # attention = self.attention(fused_out_reduced)
        out = self.refine(torch.cat((inputs0, adaptive_attention*level_f), 1))
        if self.vis:
            return out, levels_weight, adaptive_attention.sum(dim=1)
        else:
            return out


class MTLN3D(nn.Module):
    def __init__(self, vis=False):
        super(MTLN3D, self).__init__()
        # self.training = train
        self.backbone = BackBone3D()

        self.bivablock1 = BiVA(num_channels=64, first_time=True)
        # self.bivablock2 = BiVA(num_channels=64, first_time=False)

        self.ASA0 = ASA3D(level=0, vis=vis)
        self.ASA1 = ASA3D(level=1, vis=vis)
        self.ASA2 = ASA3D(level=2, vis=vis)
        self.ASA3 = ASA3D(level=3, vis=vis)

        self.fusion0 = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )

        self.fusion1 = nn.Sequential(
            nn.Conv3d(256, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )

        ### segmentaion branch
        self.attention0 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=1), nn.Sigmoid()
        )
        self.conv0 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )

        self.attention1 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=1), nn.Sigmoid()
        )
        self.conv1 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )

        self.predict_fuse0 = nn.Conv3d(64, 2, kernel_size=1)
        self.predict_fuse1 = nn.Conv3d(64, 2, kernel_size=1)

        self.predict = nn.Conv3d(64, 2, kernel_size=1)

        ### classification branch

        self.pool0 = add_conv3D(64, 64, 3, (1, 2, 2))
        self.attention2 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=1), nn.Sigmoid()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.pool1 = add_conv3D(64, 64, 3, (1, 2, 2))
        self.attention3 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=1), nn.GroupNorm(32, 64), nn.PReLU(),
            nn.Conv3d(64, 64, kernel_size=1), nn.Sigmoid()
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, 2)
        # self.fc2 = nn.Linear(4096, 64)
        # self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        layer0 = self.backbone.layer0(x)
        layer1 = self.backbone.layer1(layer0)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)
        layer4 = self.backbone.layer4(layer3)

        Scale1V, Scale2V, Scale3V, Scale4V = self.bivablock1(layer1, layer2,layer3,layer4)

        F20_upsample = F.upsample(Scale2V, size=layer1.size()[2:], mode='trilinear')
        F30_upsample = F.upsample(Scale3V, size=layer1.size()[2:], mode='trilinear')
        F40_upsample = F.upsample(Scale4V, size=layer1.size()[2:], mode='trilinear')

        fuse0 = self.fusion0(torch.cat((F40_upsample, F30_upsample, F20_upsample, Scale1V), 1))
        #
        Scale1A = self.ASA0(Scale1V, fuse0)
        Scale2A = self.ASA1(Scale2V, fuse0)
        Scale3A = self.ASA2(Scale3V, fuse0)
        Scale4A = self.ASA3(Scale4V, fuse0)

        F2_upsample = F.upsample(Scale2A, size=layer1.size()[2:], mode='trilinear')
        F3_upsample = F.upsample(Scale3A, size=layer1.size()[2:], mode='trilinear')
        F4_upsample = F.upsample(Scale4A, size=layer1.size()[2:], mode='trilinear')

        fuse1 = self.fusion1(torch.cat((F4_upsample, F3_upsample, F2_upsample, Scale1A), 1))

        ### segmentation branch
        out_F3_0 = torch.cat((Scale4A, Scale3A), 1)
        out_F3_1 = F.upsample(out_F3_0, size=Scale2A.size()[2:], mode='trilinear')

        out_F2_0 = torch.cat((out_F3_1, Scale2A), 1)
        out_F2_1 = self.conv0(self.attention0(out_F2_0) * Scale2A)
        out_F2_2 = F.upsample(out_F2_1, size=Scale1A.size()[2:], mode='trilinear')

        out_F1_0 = torch.cat((out_F2_2, Scale1A), 1)
        out_F1_1 = self.conv1(self.attention1(out_F1_0) * Scale1A)
        out_F1_2 = F.upsample(out_F1_1, size=x.size()[2:], mode='trilinear')

        ### classificication branch
        out_F10 = self.pool0(Scale1A)

        out_F20 = torch.cat((out_F10, Scale2A), 1)
        out_F21 = self.conv2(self.attention2(out_F20) * Scale2A)
        out_F22 = self.pool1(out_F21)

        out_F30 = torch.cat((out_F22, Scale3A), 1)
        out_F31 = self.conv3(self.attention3(out_F30) * Scale3A)

        out_F40 = torch.cat((out_F31, Scale4A), 1)

        fuse0 = F.upsample(fuse0, size=x.size()[2:], mode='trilinear')
        fuse1 = F.upsample(fuse1, size=x.size()[2:], mode='trilinear')

        seg_fuse0 = self.predict_fuse0(fuse0)
        seg_fuse1 = self.predict_fuse1(fuse1)

        seg_predict = self.predict(out_F1_2)

        class_predict1 = self.pool(out_F40)
        class_predict1 = class_predict1.view(class_predict1.size(0), -1)
        class_predict = self.fc(class_predict1)

        return seg_fuse0, seg_fuse1, seg_predict, class_predict
        # return predict_down11, predict_down22,predict_down32, predict_focus1, predict_focus2, predict_focus3,predict
        # return seg_predict
