import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import itertools
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from IPython import display
import matplotlib as mpl
from LapDepth.LapDepth.ska import SKA
if os.environ.get('DISPLAY', '') == '':
    mpl.use('Agg')
import matplotlib.pyplot as plt
from torch.jit import script
import geffnet


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class LKP(nn.Module):
    def __init__(self, dim, lks, sks, groups):
        super().__init__()
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU()
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, sks ** 2 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 2 * dim // groups)
        self.sks = sks
        self.groups = groups
        self.dim = dim

    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, h, width = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 2, h, width)
        return w


class LSConv(nn.Module):
    def __init__(self, dim):
        super(LSConv, self).__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=8)
        self.ska = SKA()
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.bn(self.ska(x, self.lkp(x))) + x


class WSConv2d(nn.Conv2d):
    def __init___(self, in_channels, out_channels, kernel_size, stride=1,
                  padding=0, dilation=1, groups=1, bias=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv_ws(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return WSConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                    groups=groups, bias=bias)


@script
def _mish_jit_fwd(x): return x.mul(torch.tanh(F.softplus(x)))


@script
def _mish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))


class MishJitAutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        return _mish_jit_bwd(x, grad_output)


def mish(x): return MishJitAutoFn.apply(x)


class Mish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Mish, self).__init__()

    def forward(self, x):
        return MishJitAutoFn.apply(x)


class upConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, norm, act, num_groups):
        super(upConvLayer, self).__init__()
        conv = conv_ws
        if act == 'ELU':
            act = nn.ELU()
        elif act == 'Mish':
            act = Mish()
        else:
            act = nn.ReLU(True)
        self.conv = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                         bias=False)
        if norm == 'GN':
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        else:
            self.norm = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.act = act
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        x = self.conv(x)
        return x


class myConv(nn.Module):
    def __init__(self, in_ch, out_ch, kSize, stride=1,
                 padding=0, dilation=1, bias=True, norm='GN', act='ELU', num_groups=32):
        super(myConv, self).__init__()
        conv = conv_ws
        if act == 'ELU':
            act = nn.ELU()
        elif act == 'Mish':
            act = Mish()
        else:
            act = nn.ReLU(True)
        module = []
        if norm == 'GN':
            module.append(nn.GroupNorm(num_groups=num_groups, num_channels=in_ch))
        else:
            module.append(nn.BatchNorm2d(in_ch, eps=0.001, momentum=0.1, affine=True, track_running_stats=True))
        module.append(act)
        module.append(conv(in_ch, out_ch, kernel_size=kSize, stride=stride,
                           padding=padding, dilation=dilation, groups=1, bias=bias))
        self.module = nn.Sequential(*module)

    def forward(self, x):
        out = self.module(x)
        return out


class deepFeatureExtractor_ResNext101(nn.Module):
    def __init__(self, args, lv6=False):
        super(deepFeatureExtractor_ResNext101, self).__init__()
        self.args = args
        self.encoder = models.resnext101_32x8d(pretrained=True)
        self.fixList = ['layer1.0', 'layer1.1', '.bn']
        self.lv6 = lv6
        if lv6 is True:
            self.layerList = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.dimList = [64, 256, 512, 1024, 2048]
        else:
            del self.encoder.layer4
            del self.encoder.fc
            self.layerList = ['relu', 'layer1', 'layer2', 'layer3']
            self.dimList = [64, 256, 512, 1024]
        for name, parameters in self.encoder.named_parameters():
            if name == 'conv1.weight':
                parameters.requires_grad = False
            if any(x in name for x in self.fixList):
                parameters.requires_grad = False

    def forward(self, x):
        out_featList = []
        feature = x
        for k, v in self.encoder._modules.items():
            if k == 'avgpool':
                break
            feature = v(feature)
            if any(x in k for x in self.layerList):
                out_featList.append(feature)
        return out_featList

    def freeze_bn(self, enable=False):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()
                module.weight.requires_grad = enable
                module.bias.requires_grad = enable


class Dilated_bottleNeck(nn.Module):
    def __init__(self, norm, act, in_feat):
        super(Dilated_bottleNeck, self).__init__()
        conv = conv_ws
        self.reduction1 = conv(in_feat, in_feat // 2, kernel_size=1, stride=1, bias=False, padding=0)
        self.aspp_d3 = nn.Sequential(
            myConv(in_feat // 2, in_feat // 4, kSize=1, stride=1, padding=0, dilation=1, bias=False, norm=norm, act=act,
                   num_groups=(in_feat // 2) // 16),
            myConv(in_feat // 4, in_feat // 4, kSize=3, stride=1, padding=3, dilation=3, bias=False, norm=norm, act=act,
                   num_groups=(in_feat // 4) // 16))
        self.aspp_d6 = nn.Sequential(
            myConv(in_feat // 2 + in_feat // 4, in_feat // 4, kSize=1, stride=1, padding=0, dilation=1, bias=False,
                   norm=norm, act=act, num_groups=(in_feat // 2 + in_feat // 4) // 16),
            myConv(in_feat // 4, in_feat // 4, kSize=3, stride=1, padding=6, dilation=6, bias=False, norm=norm, act=act,
                   num_groups=(in_feat // 4) // 16))
        self.aspp_d12 = nn.Sequential(
            myConv(in_feat, in_feat // 4, kSize=1, stride=1, padding=0, dilation=1, bias=False, norm=norm, act=act,
                   num_groups=(in_feat) // 16),
            myConv(in_feat // 4, in_feat // 4, kSize=3, stride=1, padding=12, dilation=12, bias=False, norm=norm,
                   act=act, num_groups=(in_feat // 4) // 16))
        self.aspp_d18 = nn.Sequential(
            myConv(in_feat + in_feat // 4, in_feat // 4, kSize=1, stride=1, padding=0, dilation=1, bias=False,
                   norm=norm, act=act, num_groups=(in_feat + in_feat // 4) // 16),
            myConv(in_feat // 4, in_feat // 4, kSize=3, stride=1, padding=18, dilation=18, bias=False, norm=norm,
                   act=act, num_groups=(in_feat // 4) // 16))
        self.reduction2 = myConv(((in_feat // 4) * 4) + (in_feat // 2), in_feat // 2, kSize=3, stride=1, padding=1,
                                 bias=False, norm=norm, act=act, num_groups=((in_feat // 4) * 4 + (in_feat // 2)) // 16)

    def forward(self, x):
        x = self.reduction1(x)
        d3 = self.aspp_d3(x)
        cat1 = torch.cat([x, d3], dim=1)
        d6 = self.aspp_d6(cat1)
        cat2 = torch.cat([cat1, d6], dim=1)
        d12 = self.aspp_d12(cat2)
        cat3 = torch.cat([cat2, d12], dim=1)
        d18 = self.aspp_d18(cat3)
        out = self.reduction2(torch.cat([x, d3, d6, d12, d18], dim=1))
        return out


class Lap_decoder_lv5(nn.Module):
    def __init__(self, args, dimList):
        super(Lap_decoder_lv5, self).__init__()
        norm = args.norm
        conv = conv_ws
        if norm == 'GN':
            if args.rank == 0:
                print("==> Norm: GN")
        else:
            if args.rank == 0:
                print("==> Norm: BN")
        if args.act == 'ELU':
            act = 'ELU'
        elif args.act == 'Mish':
            act = 'Mish'
        else:
            act = 'ReLU'
        kSize = 3
        self.max_depth = args.max_depth
        self.ASPP = Dilated_bottleNeck(norm, act, dimList[3])
        self.dimList = dimList

        # Pyramid Level 5
        self.decoder1 = nn.Sequential(
            LSConv(dimList[3] // 2),
            myConv(dimList[3] // 2, dimList[3] // 4, kSize, stride=1, padding=kSize // 2, bias=False,
                   norm=norm, act=act, num_groups=(dimList[3] // 2) // 16),
            myConv(dimList[3] // 4, dimList[3] // 8, kSize, stride=1, padding=kSize // 2, bias=False,
                   norm=norm, act=act, num_groups=(dimList[3] // 4) // 16),
            myConv(dimList[3] // 8, dimList[3] // 16, kSize, stride=1, padding=kSize // 2, bias=False,
                   norm=norm, act=act, num_groups=(dimList[3] // 8) // 16),
            myConv(dimList[3] // 16, dimList[3] // 32, kSize, stride=1, padding=kSize // 2, bias=False,
                   norm=norm, act=act, num_groups=(dimList[3] // 16) // 16),
            myConv(dimList[3] // 32, 1, kSize, stride=1, padding=kSize // 2, bias=False,
                   norm=norm, act=act, num_groups=(dimList[3] // 32) // 16)
        )

        # Pyramid Level 4
        self.decoder2_up1 = upConvLayer(dimList[3] // 2, dimList[3] // 4, 2, norm, act, (dimList[3] // 2) // 16)
        self.decoder2_reduc1 = myConv(dimList[3] // 4 + dimList[2], dimList[3] // 4 - 4, kSize=1, stride=1, padding=0,
                                      bias=False, norm=norm, act=act, num_groups=(dimList[3] // 4 + dimList[2]) // 16)
        self.decoder2_1 = LSConv(dimList[3] // 4)
        self.decoder2_2 = myConv(dimList[3] // 4, dimList[3] // 8, kSize, stride=1, padding=kSize // 2, bias=False,
                                 norm=norm, act=act, num_groups=(dimList[3] // 4) // 16)
        self.decoder2_3 = myConv(dimList[3] // 8, dimList[3] // 16, kSize, stride=1, padding=kSize // 2, bias=False,
                                 norm=norm, act=act, num_groups=(dimList[3] // 8) // 16)
        self.decoder2_4 = myConv(dimList[3] // 16, 1, kSize, stride=1, padding=kSize // 2, bias=False,
                                 norm=norm, act=act, num_groups=(dimList[3] // 16) // 16)

        # Pyramid Level 3
        self.decoder2_1_up2 = upConvLayer(dimList[3] // 4, dimList[3] // 8, 2, norm, act, (dimList[3] // 4) // 16)
        self.decoder2_1_reduc2 = myConv(dimList[3] // 8 + dimList[1], dimList[3] // 8 - 4, kSize=1, stride=1, padding=0,
                                        bias=False, norm=norm, act=act, num_groups=(dimList[3] // 8 + dimList[1]) // 16)
        self.decoder2_1_1 = LSConv(dimList[3] // 8)
        self.decoder2_1_2 = myConv(dimList[3] // 8, dimList[3] // 16, kSize, stride=1, padding=kSize // 2, bias=False,
                                   norm=norm, act=act, num_groups=(dimList[3] // 8) // 16)
        self.decoder2_1_3 = myConv(dimList[3] // 16, 1, kSize, stride=1, padding=kSize // 2, bias=False,
                                   norm=norm, act=act, num_groups=(dimList[3] // 16) // 16)

        # Pyramid Level 2
        self.decoder2_1_1_up3 = upConvLayer(dimList[3] // 8, dimList[3] // 16, 2, norm, act, (dimList[3] // 8) // 16)
        self.decoder2_1_1_reduc3 = myConv(dimList[3] // 16 + dimList[0], dimList[3] // 16 - 4, kSize=1, stride=1,
                                          padding=0, bias=False, norm=norm, act=act,
                                          num_groups=(dimList[3] // 16 + dimList[0]) // 16)
        self.decoder2_1_1_1 = myConv(dimList[3] // 16, dimList[3] // 16, kSize, stride=1, padding=kSize // 2,
                                     bias=False, norm=norm, act=act, num_groups=(dimList[3] // 16) // 16)
        self.decoder2_1_1_2 = myConv(dimList[3] // 16, 1, kSize, stride=1, padding=kSize // 2, bias=False,
                                     norm=norm, act=act, num_groups=(dimList[3] // 16) // 16)

        # Pyramid Level 1
        self.decoder2_1_1_1_up4 = upConvLayer(dimList[3] // 16, dimList[3] // 16 - 4, 2, norm, act,
                                              (dimList[3] // 16) // 16)
        self.decoder2_1_1_1_1 = myConv(dimList[3] // 16, dimList[3] // 16, kSize, stride=1, padding=kSize // 2,
                                       bias=False, norm=norm, act=act, num_groups=(dimList[3] // 16) // 16)
        self.decoder2_1_1_1_2 = myConv(dimList[3] // 16, dimList[3] // 32, kSize, stride=1, padding=kSize // 2,
                                       bias=False, norm=norm, act=act, num_groups=(dimList[3] // 16) // 16)
        self.decoder2_1_1_1_3 = myConv(dimList[3] // 32, 1, kSize, stride=1, padding=kSize // 2, bias=False,
                                       norm=norm, act=act, num_groups=(dimList[3] // 32) // 16)

        self.upscale = F.interpolate

    def forward(self, x, rgb):
        cat1, cat2, cat3, dense_feat = x[0], x[1], x[2], x[3]
        rgb_lv6, rgb_lv5, rgb_lv4, rgb_lv3, rgb_lv2, rgb_lv1 = rgb[0], rgb[1], rgb[2], rgb[3], rgb[4], rgb[5]
        dense_feat = self.ASPP(dense_feat)

        # Level 5
        lap_lv5 = torch.sigmoid(self.decoder1(dense_feat))
        lap_lv5_up = self.upscale(lap_lv5, scale_factor=2, mode='bilinear')

        # Level 4
        dec2 = self.decoder2_up1(dense_feat)
        dec2 = self.decoder2_reduc1(torch.cat([dec2, cat3], dim=1))
        dec2_up = self.decoder2_1(torch.cat([dec2, lap_lv5_up, rgb_lv4], dim=1))
        dec2 = self.decoder2_2(dec2_up)
        dec2 = self.decoder2_3(dec2)
        lap_lv4 = torch.tanh(self.decoder2_4(dec2) + (0.1 * rgb_lv4.mean(dim=1, keepdim=True)))
        lap_lv4_up = self.upscale(lap_lv4, scale_factor=2, mode='bilinear')

        # Level 3
        dec3 = self.decoder2_1_up2(dec2_up)
        dec3 = self.decoder2_1_reduc2(torch.cat([dec3, cat2], dim=1))
        dec3_up = self.decoder2_1_1(torch.cat([dec3, lap_lv4_up, rgb_lv3], dim=1))
        dec3 = self.decoder2_1_2(dec3_up)
        lap_lv3 = torch.tanh(self.decoder2_1_3(dec3) + (0.1 * rgb_lv3.mean(dim=1, keepdim=True)))
        lap_lv3_up = self.upscale(lap_lv3, scale_factor=2, mode='bilinear')

        # Level 2
        dec4 = self.decoder2_1_1_up3(dec3_up)
        dec4 = self.decoder2_1_1_reduc3(torch.cat([dec4, cat1], dim=1))
        dec4_up = self.decoder2_1_1_1(torch.cat([dec4, lap_lv3_up, rgb_lv2], dim=1))
        lap_lv2 = torch.tanh(self.decoder2_1_1_2(dec4_up) + (0.1 * rgb_lv2.mean(dim=1, keepdim=True)))
        lap_lv2_up = self.upscale(lap_lv2, scale_factor=2, mode='bilinear')

        # Level 1
        dec5 = self.decoder2_1_1_1_up4(dec4_up)
        dec5 = self.decoder2_1_1_1_1(torch.cat([dec5, lap_lv2_up, rgb_lv1], dim=1))
        dec5 = self.decoder2_1_1_1_2(dec5)
        lap_lv1 = torch.tanh(self.decoder2_1_1_1_3(dec5) + (0.1 * rgb_lv1.mean(dim=1, keepdim=True)))

        # Laplacian restoration
        lap_lv4_img = lap_lv4 + lap_lv5_up
        lap_lv3_img = lap_lv3 + self.upscale(lap_lv4_img, scale_factor=2, mode='bilinear')
        lap_lv2_img = lap_lv2 + self.upscale(lap_lv3_img, scale_factor=2, mode='bilinear')
        final_depth = lap_lv1 + self.upscale(lap_lv2_img, scale_factor=2, mode='bilinear')
        final_depth = torch.sigmoid(final_depth)

        return [(lap_lv5) * self.max_depth, (lap_lv4) * self.max_depth, (lap_lv3) * self.max_depth,
                (lap_lv2) * self.max_depth, (lap_lv1) * self.max_depth], final_depth * self.max_depth


class LS(nn.Module):
    def __init__(self, args):
        super(LS, self).__init__()
        lv6 = args.lv6
        encoder = args.encoder
        if encoder == 'ResNext101':
            self.encoder = deepFeatureExtractor_ResNext101(args, lv6)
            self.decoder = Lap_decoder_lv5(args, self.encoder.dimList)

    def forward(self, x):
        out_featList = self.encoder(x)
        rgb_down2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        rgb_down4 = F.interpolate(rgb_down2, scale_factor=0.5, mode='bilinear')
        rgb_down8 = F.interpolate(rgb_down4, scale_factor=0.5, mode='bilinear')
        rgb_down16 = F.interpolate(rgb_down8, scale_factor=0.5, mode='bilinear')
        rgb_down32 = F.interpolate(rgb_down16, scale_factor=0.5, mode='bilinear')
        rgb_up16 = F.interpolate(rgb_down32, rgb_down16.shape[2:], mode='bilinear')
        rgb_up8 = F.interpolate(rgb_down16, rgb_down8.shape[2:], mode='bilinear')
        rgb_up4 = F.interpolate(rgb_down8, rgb_down4.shape[2:], mode='bilinear')
        rgb_up2 = F.interpolate(rgb_down4, rgb_down2.shape[2:], mode='bilinear')
        rgb_up = F.interpolate(rgb_down2, x.shape[2:], mode='bilinear')
        lap1 = x - rgb_up
        lap2 = rgb_down2 - rgb_up2
        lap3 = rgb_down4 - rgb_up4
        lap4 = rgb_down8 - rgb_up8
        lap5 = rgb_down16 - rgb_up16
        rgb_list = [rgb_down32, lap5, lap4, lap3, lap2, lap1]
        d_res_list, depth = self.decoder(out_featList, rgb_list)
        return d_res_list, depth

    def train(self, mode=True):
        super().train(mode)
        self.encoder.freeze_bn()