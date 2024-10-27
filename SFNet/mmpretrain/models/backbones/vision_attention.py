# -*- coding:utf-8 -*-
''' Visual attention mechanism.

Version 1.0  2024-04-11 11:47:39
by QiJi Refence:
- https://github.com/Jongchan/attention-module/tree/master/MODELS

TODO:
1. https://github.com/cfzd/FcaNet

'''
import torch
import torch.nn as nn
from torch.nn import functional as F


class ChannelAttention(nn.Module):
    """ Channel Attention from [CBAM: Convolutional Block Attention Module (ECCV2018)](https://arxiv.org/abs/1807.06521).
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes//ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes//ratio, in_planes, 1, bias=False),
        )

    def forward(self, x):
        x_avg = self.shared_MLP(self.avg_pool(x))
        x_max = self.shared_MLP(self.max_pool(x))
        scale = torch.sigmoid(x_avg + x_max)
        return x * scale


class SpatialAttention(nn.Module):
    """ Spatial Attention from [CBAM: Convolutional Block Attention Module (ECCV2018)](https://arxiv.org/abs/1807.06521).
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size-1) // 2, bias=False)

    def channel_pool(self, x):
        return torch.cat(
            [torch.max(x, dim=1, keepdim=True)[0], torch.mean(x, dim=1, keepdim=True)], dim=1
        )

    def forward(self, x):
        x_compress = self.channel_pool(x)
        x_out = self.conv(x_compress)

        scale = torch.sigmoid(x_out)
        return x * scale


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class SpatialGate(nn.Module):
    """ Spatial Attention from [BAM: Bottleneck Attention Module (BMVC2018)](https://arxiv.org/abs/1807.06514).
    """
    def __init__(
        self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4
    ):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module(
            "gate_s_conv_reduce0",
            nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1),
        )
        self.gate_s.add_module(
            "gate_s_bn_reduce0", nn.BatchNorm2d(gate_channel // reduction_ratio)
        )
        self.gate_s.add_module("gate_s_relu_reduce0", nn.ReLU())
        for i in range(dilation_conv_num):
            self.gate_s.add_module(
                "gate_s_conv_di_%d" % i,
                nn.Conv2d(
                    gate_channel // reduction_ratio,
                    gate_channel // reduction_ratio,
                    kernel_size=3,
                    padding=dilation_val,
                    dilation=dilation_val,
                ),
            )
            self.gate_s.add_module(
                "gate_s_bn_di_%d" % i, nn.BatchNorm2d(gate_channel // reduction_ratio)
            )
            self.gate_s.add_module("gate_s_relu_di_%d" % i, nn.ReLU())
        self.gate_s.add_module(
            "gate_s_conv_final",
            nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1),
        )

    def forward(self, x):
        scale = self.gate_s(x).expand_as(x)
        return x * scale


class ChannelGate(nn.Module):
    """ Channel Attention from [BAM: Bottleneck Attention Module (BMVC2018)](https://arxiv.org/abs/1807.06514).
    """
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module("flatten", Flatten())
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module(
                "gate_c_fc_%d" % i, nn.Linear(gate_channels[i], gate_channels[i + 1])
            )
            self.gate_c.add_module(
                "gate_c_bn_%d" % (i + 1), nn.BatchNorm1d(gate_channels[i + 1])
            )
            self.gate_c.add_module("gate_c_relu_%d" % (i + 1), nn.ReLU())
        self.gate_c.add_module(
            "gate_c_fc_final", nn.Linear(gate_channels[-2], gate_channels[-1])
        )

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, x.size(2), stride=x.size(2))
        scale = 1 + torch.sigmoid(self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(x))
        return x * scale
