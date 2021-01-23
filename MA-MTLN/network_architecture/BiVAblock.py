from torch import nn
import torch
import torch.nn.functional as F

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class VAblock(nn.Module):
    """
    created by zyt
    """
    def __init__(self, num_channels, level, cardinality):
        super(VAblock, self).__init__()
        self.level = level
        if level == 1:
            self.l1_va_branch0 = nn.Sequential(nn.Conv3d(num_channels,num_channels,kernel_size=1),
                                               nn.Conv3d(num_channels,num_channels,kernel_size=3, stride=1, padding=1,dilation=1), nn.Sigmoid())
            self.l1_va_branch1 = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=3,padding=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1,padding=3, dilation=3), nn.Sigmoid())
            self.l1_va_branch2 = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=3,padding=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1,padding=5,dilation=5), nn.Sigmoid())
            self.l1_va_branch3 = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding=(5, 7, 7),dilation=(5, 7, 7)), nn.Sigmoid())
        elif level == 2:
            self.l2_va_branch0 = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding=1,dilation=1), nn.Sigmoid())
            self.l2_va_branch1 = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding=3,dilation=3), nn.Sigmoid())
            self.l2_va_branch2 = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding=5, dilation=5), nn.Sigmoid())
        elif level == 3:
            self.l3_va_branch0 = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, dilation=1), nn.Sigmoid())
            self.l3_va_branch1 = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=3,padding=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding=3, dilation=3), nn.Sigmoid())
        elif level == 4:
            self.l4_va_branch0 = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, dilation=1), nn.Sigmoid())
            self.l4_va_branch1 = nn.Sequential(nn.Conv3d(num_channels, num_channels, kernel_size=3, padding=1),
                                               nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding=3, dilation=3), nn.Sigmoid())
        self.fusion1 = nn.Sequential(
            nn.Conv3d(256, num_channels, kernel_size=1)
            # nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            # nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.fusion2 = nn.Sequential(
            nn.Conv3d(192, num_channels, kernel_size=1)
            # nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            # nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.fusion3 = nn.Sequential(
            nn.Conv3d(128, num_channels, kernel_size=1)
            # nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            # nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.fusion4 = nn.Sequential(
            nn.Conv3d(128, num_channels, kernel_size=1)
            # nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU(),
            # nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.GroupNorm(32, 64), nn.PReLU()
        )
        # self.gn = nn.GroupNorm(32, 64)
        self.PReLU = nn.PReLU()

    def forward(self, x):
        if self.level == 1:
            x_l1_branch0 = self.l1_va_branch0(x) * x
            x_l1_branch1 = self.l1_va_branch1(x) * x
            x_l1_branch2 = self.l1_va_branch2(x) * x
            x_l1_branch3 = self.l1_va_branch3(x) * x
            x_out1 = self.fusion1(torch.cat((x_l1_branch0,x_l1_branch1,x_l1_branch2,x_l1_branch3),1))
            out = self.PReLU(x_out1 + x)

        elif self.level == 2:
            x_l2_branch0 = self.l2_va_branch0(x) * x
            x_l2_branch1 = self.l2_va_branch1(x) * x
            x_l2_branch2 = self.l2_va_branch2(x) * x
            x_out2 = self.fusion2(torch.cat((x_l2_branch0, x_l2_branch1, x_l2_branch2), 1))
            out = self.PReLU(x_out2 + x)

        elif self.level == 3:
            x_l3_branch0 = self.l3_va_branch0(x) * x
            x_l3_branch1 = self.l3_va_branch1(x) * x
            x_out3 = self.fusion3(torch.cat((x_l3_branch0, x_l3_branch1), 1))
            out = self.PReLU(x_out3 + x)

        elif self.level == 4:
            x_l4_branch0 = self.l4_va_branch0(x) * x
            x_l4_branch1 = self.l4_va_branch1(x) * x
            # print(x.size(),x_l4_branch0.size(),x_l4_branch1.size())
            x_out4 = self.fusion4(torch.cat((x_l4_branch0, x_l4_branch1), 1))
            out = self.PReLU(x_out4 + x)

        return out

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

class BiVA(nn.Module):
    """
    created by zyt
    """
    def __init__(self,num_channels, epsilon=1e-4, first_time=False, onnx_export=False, attention=True):
        super(BiVA,self).__init__()
        self.epsilon = epsilon
        self.first_time = first_time
        #conv layers
        self.conv_l1_down_channel = nn.Sequential(
            nn.Conv3d(128, num_channels, kernel_size=1),
            nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.conv_l2_down_channel = nn.Sequential(
            nn.Conv3d(256, num_channels, kernel_size=1),
            nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.conv_l3_down_channel = nn.Sequential(
            nn.Conv3d(512, num_channels, kernel_size=1),
            nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.conv_l4_down_channel = nn.Sequential(
            nn.Conv3d(512, num_channels, kernel_size=1),
            nn.GroupNorm(32, 64), nn.PReLU()
        )
        self.VAblock1 = VAblock(num_channels, level=1, cardinality=32)
        self.VAblock2 = VAblock(num_channels, level=2, cardinality=32)
        self.VAblock3 = VAblock(num_channels, level=3, cardinality=32)
        self.VAblock4 = VAblock(num_channels, level=4, cardinality=32)

        self.pool1 = add_conv3D(num_channels, num_channels, 1, (1, 2, 2))
        self.pool2 = add_conv3D(num_channels, num_channels, 1, (1, 2, 2))

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        # self.swish1 = MemoryEfficientSwish() if not onnx_export else Swish()
        # self.swish2 = MemoryEfficientSwish() if not onnx_export else Swish()
        # self.swish3 = MemoryEfficientSwish() if not onnx_export else Swish()
        # self.swish4 = MemoryEfficientSwish() if not onnx_export else Swish()

        # Weight
        self.F3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.F3_w1_relu = nn.PReLU()
        self.F2_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.F2_w1_relu = nn.PReLU()
        self.F1_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.F1_w1_relu = nn.PReLU()

        self.F1_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.F1_w2_relu = nn.PReLU()
        self.F2_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.F2_w2_relu = nn.PReLU()
        self.F3_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.F3_w2_relu = nn.PReLU()
        self.F4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.F4_w2_relu = nn.PReLU()


        self.attention = attention

    def forward(self, inputs1, inputs2, inputs3, inputs4):
        """
        illustration of a minimal bifpn unit
            F4_0 ------------>F4_1-------------> F4_2 -------->
               |-------------|                ↑
                             ↓                |
            F3_0 ---------> F3_1 ---------> F3_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            F2_0 ---------> F2_1 ---------> F2_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            F1_0 ---------> F1_1 ---------> F1_2 -------->
        """
        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation

        if self.attention:
            F1_out, F2_out, F3_out, F4_out = self._forward_fast_attention(inputs1, inputs2, inputs3, inputs4)
        else:
            F1_out, F2_out, F3_out, F4_out = self._forward(inputs1, inputs2, inputs3, inputs4)

        return F1_out, F2_out, F3_out, F4_out

    def _forward_fast_attention(self, inputs1, inputs2, inputs3, inputs4):

        if self.first_time:
            # Down channel
            inputs1 = self.conv_l1_down_channel(inputs1)
            inputs2 = self.conv_l2_down_channel(inputs2)
            inputs3 = self.conv_l3_down_channel(inputs3)
            inputs4 = self.conv_l4_down_channel(inputs4)
        else:
            inputs1=inputs1
            inputs2=inputs2
            inputs3=inputs3
            inputs4=inputs4

        # F4_0 to F4_1
        F4_1 = self.VAblock4(inputs4)

        # Weights for F3_0 and F4_1 to F3_1
        F3_w1 = self.F3_w1_relu(self.F3_w1)
        weight = F3_w1 / (torch.sum(F3_w1, dim=0) + self.epsilon)
        # Connections for F3_0 and F4_1 to F3_1 respectively
        F3_1 = self.VAblock3(self.swish(weight[0] * inputs3 + weight[1] * F4_1))

        # Weights for F2_0 and F3_1 to F2_1
        F2_w1 = self.F2_w1_relu(self.F2_w1)
        weight = F2_w1 / (torch.sum(F2_w1, dim=0) + self.epsilon)
        # Connections for F2_0 and F3_1 to F2_1 respectively
        F3_1_1 = F.upsample(F3_1, size=inputs2.size()[2:], mode='trilinear')
        F2_1 = self.VAblock2(self.swish(weight[0] * inputs2 + weight[1] * F3_1_1))

        # Weights for F1_0 and F2_1 to F1_1
        F1_w1 = self.F1_w1_relu(self.F1_w1)
        weight = F1_w1 / (torch.sum(F1_w1, dim=0) + self.epsilon)
        # Connections for F1_0 and F2_1 to F1_1 respectively
        F2_1_1 = F.upsample(F2_1, size=inputs1.size()[2:], mode='trilinear')
        F1_1 = self.VAblock1(self.swish(weight[0] * inputs1 + weight[1] * F2_1_1))


        # Weights for F1_0, F1_1 to F1_2
        F1_w2 = self.F1_w2_relu(self.F1_w2)
        weight = F1_w2 / (torch.sum(F1_w2, dim=0) + self.epsilon)
        # Connections for F1_0 and F1_1 to F1_2 respectively
        F1_2 = self.swish(weight[0] * inputs1 + weight[1] * F1_1)

        # Weights for F2_0, F2_1 and F1_2 to F2_2
        F2_w2 = self.F2_w2_relu(self.F2_w2)
        weight = F2_w2 / (torch.sum(F2_w2, dim=0) + self.epsilon)
        # Connections for F2_0, F2_1 and F1_2 to F2_2 respectively
        F1_2_1 = self.pool1(F1_2)
        F2_2 = self.swish(weight[0] * inputs2 + weight[1] * F2_1 + weight[2] * F1_2_1)

        # Weights for F3_0, F3_1 and F2_2 to F3_2
        F3_w2 = self.F3_w2_relu(self.F3_w2)
        weight = F3_w2 / (torch.sum(F3_w2, dim=0) + self.epsilon)
        # Connections for F3_0, F3_1 and F2_2 to F3_2 respectively
        F2_2_1 = self.pool2(F2_2)
        F3_2 = self.swish(weight[0] * inputs3 + weight[1] * F3_1 + weight[2] * F2_2_1)

        # Weights for F4_0 , F4_1 and F3_2 to F4_2
        F4_w2 = self.F4_w2_relu(self.F4_w2)
        weight = F4_w2 / (torch.sum(F4_w2, dim=0) + self.epsilon)
        # Connections for F4_0, F4_1 and F3_2 to F4_2
        F4_2 = self.swish(weight[0] * inputs4 + weight[1] * F4_1 + weight[2] * F3_2)

        return F1_2, F2_2, F3_2, F4_2

    def _forward(self, inputs1, inputs2, inputs3, inputs4):
        if self.first_time:
            # Down channel
            inputs1 = self.conv_l1_down_channel(inputs1)
            inputs2 = self.conv_l2_down_channel(inputs2)
            inputs3 = self.conv_l3_down_channel(inputs3)
            inputs4 = self.conv_l4_down_channel(inputs4)
        else:
            inputs1 = inputs1
            inputs2 = inputs2
            inputs3 = inputs3
            inputs4 = inputs4

        F4_1 = self.VAblock4(inputs4)

        # Connections for F3_0 and F4_1 to F3_1 respectively
        F3_1 = self.VAblock3(self.swish(inputs3 + F4_1))

        # Connections for F2_0 and F3_1 to F2_1 respectively
        F3_1_1 = F.upsample(F3_1, size=inputs2.size()[2:], mode='trilinear')
        F2_1 = self.VAblock2(self.swish(inputs2 + F3_1_1))

        # Connections for F1_0 and F2_1 to F1_1 respectively
        F2_1_1 = F.upsample(F2_1, size=inputs1.size()[2:], mode='trilinear')
        F1_1 = self.VAblock1(self.swish(inputs1 + F2_1_1))

        # Connections for F1_0 and F1_1 to F1_2 respectively
        F1_2 = self.swish(inputs1 + F1_1)

        # Connections for F2_0, F2_1 and F1_2 to F2_2 respectively
        F1_2_1 = self.pool1(F1_2)
        F2_2 = self.swish(inputs2 + F2_1 + F1_2_1)

        # Connections for F3_0, F3_1 and F2_2 to F3_2 respectively
        F2_2_1 = self.pool2(F2_2)
        F3_2 = self.swish(inputs3 + F3_1 + F2_2_1)

        # Connections for F4_0, F4_1 and F3_2 to F4_2
        F4_2 = self.swish(inputs4 + F4_1 + F3_2)

        return F1_2, F2_2, F3_2, F4_2

