import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

def softmax_helper(x):

    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn


class JILoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """

        """
        super(JILoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        dc = (tp + self.smooth) / (tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class JI_and_Focal_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, aggregate="sum"):
        super(JI_and_Focal_loss, self).__init__()
        self.aggregate = aggregate
        self.JI = JILoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        self.Focal = FocalLoss(gamma=1.0, alpha=0.2, size_average=False)

    def forward(self, net_output, target):
        JI_loss = self.JI(net_output, target)
        Focal_loss = self.Focal(net_output, target)
        if self.aggregate == "sum":
            result = JI_loss + 0.1*Focal_loss
            # result = dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum.cuda()

class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        target = target.long()
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,D,H,W => N,C,D*H*W
            input = input.transpose(1, 2)    # N,C,D*H*W => N,D*H*W,C
            input = input.contiguous().view(-1, input.size()[2])  # N,D*H*W,C => N*D*H*W,C
            target = target.view(-1, 1)

            logpt = F.log_softmax(input)
            logpt = logpt.gather(1, target)
            logpt = logpt.view(-1)
            pt = Variable(logpt.data.exp())

            if self.alpha is not None:
                if self.alpha.type() != input.data.type():
                    self.alpha = self.alpha.type_as(input.data)
                at = self.alpha.gather(0, target.data.view(-1))
                logpt = logpt * Variable(at)

            loss = -(1 * (1 - pt) ** self.gamma) * logpt


            if self.size_average:
                return loss.mean()
            else:
                return loss.sum()

def dice_ratio(seg, gt):
    """
    define the dice ratio
    :param seg: segmentation result
    :param gt: ground truth
    :return:
    """
    seg = seg.flatten()
    seg[seg > 0.5] = 1
    seg[seg <= 0.5] = 0

    gt = gt.flatten()
    gt[gt > 0.5] = 1
    gt[gt <= 0.5] = 0

    same = (seg * gt).sum()

    dice = 2*float(same)/float(gt.sum() + seg.sum())

    return dice