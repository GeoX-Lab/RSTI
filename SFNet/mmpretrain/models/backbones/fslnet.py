# -*- coding:utf-8 -*-
"""
Modified from the FSLNet proposed in paper "Joint Learning of Frequency and Spatial Domains for Dense Image Prediction" by Shaocheng JIA and Wei YAO [Article](https://www.sciencedirect.com/science/article/abs/pii/S092427162200288X?via%3Dihub).

"""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS
from .base_backbone import BaseBackbone
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

import numpy as np
import torch
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
import torch.fft as fft


padding_mode = "reflect"
# reflect, replicate,,zeros
predim = 64


def upsample(x):
    """Upsample input tensor by a factor of 2"""
    return F.interpolate(x, scale_factor=2, mode="nearest")


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        if isinstance(m, nn.LayerNorm) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def RGB2MP(imgs):
    fft_imgs = fft.rfftn(imgs, dim=(2, 3), norm="ortho")
    r = torch.real(fft_imgs)
    i = torch.imag(fft_imgs)
    return torch.cat([r, i], dim=1)


def MP2Disp(MP_map):
    _, c, _, _ = MP_map.shape
    r = MP_map[:, 0 : c // 2, :, :]
    i = MP_map[:, c // 2 :, :, :]
    MP_map_complex = r + (1j * i)
    rimg = fft.irfftn(MP_map_complex, dim=(2, 3), norm="ortho")
    return rimg


class WaveTransform(BaseModule):
    """Layer to perform wave transform"""

    def __init__(self, in_channel, out_channel, h, w):
        super(WaveTransform, self).__init__()
        # self.weights = torch.nn.Parameter(torch.rand(1, 1, h, w) * 0.02)
        self.multiheads = nn.Conv2d(in_channel, out_channel, 1, bias=False, groups=1)

    def forward(self, x):
        x = self.multiheads(x)
        return x


class Act(BaseModule):
    def __init__(self):
        super(Act, self).__init__()
        self.act = nn.SiLU(True)

    def forward(self, x):
        return self.act(x)


class WaveTransformBlock(BaseModule):
    """Layer to perform wave transform"""

    def __init__(self, in_channel, out_channel, h, w):
        super(WaveTransformBlock, self).__init__()
        self.wavet = WaveTransform(in_channel, out_channel, h, w)
        self.act = Act()
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.wavet(x)
        x = self.act(x)
        x = self.bn(x)
        return x


class CNNBlock(BaseModule):
    def __init__(self, in_channels, out_channels, k=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=k,
                padding=k // 2,
                stride=stride,
                bias=False,
                padding_mode=padding_mode,
            ),
            Act(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.net(x)


class FLBlock(BaseModule):
    def __init__(self, in_channels, out_channels, h, w, mode="encoder", predim=predim):
        super().__init__()

        self.net_global = nn.Sequential(
            WaveTransformBlock(in_channels, out_channels, h, w),
            WaveTransformBlock(out_channels, out_channels, h, w),
        )

        if in_channels // 2 <= predim and out_channels // 2 <= predim:
            self.net_local = nn.Sequential(
                CNNBlock(in_channels // 2, out_channels // 2),
                CNNBlock(out_channels // 2, out_channels // 2),
                CNNBlock(out_channels // 2, out_channels // 2),
                CNNBlock(out_channels // 2, out_channels // 2),
            )

        if in_channels // 2 > predim and out_channels // 2 <= predim:
            self.net_local = nn.Sequential(
                CNNBlock(in_channels // 2, predim, k=1),
                CNNBlock(predim, out_channels // 2),
                CNNBlock(out_channels // 2, out_channels // 2),
                CNNBlock(out_channels // 2, out_channels // 2),
                CNNBlock(out_channels // 2, out_channels // 2),
            )

        if in_channels // 2 > predim and out_channels // 2 > predim:
            self.net_local = nn.Sequential(
                CNNBlock(in_channels // 2, predim, k=1),
                CNNBlock(predim, predim),
                CNNBlock(predim, predim),
                CNNBlock(predim, predim),
                CNNBlock(predim, predim),
                CNNBlock(predim, out_channels // 2, k=1),
            )

        if in_channels // 2 <= predim and out_channels // 2 > predim:
            self.net_local = nn.Sequential(
                CNNBlock(in_channels // 2, predim),
                CNNBlock(predim, predim),
                CNNBlock(predim, predim),
                CNNBlock(predim, predim),
                CNNBlock(predim, out_channels // 2, k=1),
            )

        self.compression = nn.Sequential(
            CNNBlock(out_channels, out_channels // 2),
        )

        self.mode = mode
        if mode == "encoder":
            self.resulotionchange = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        b, _, h, w = x.shape
        img = x

        x = RGB2MP(x)
        x = self.net_global(x)
        x = MP2Disp(x)
        if x.size(2) != h or x.size(3) != w:
            x = F.interpolate(x, [h, w], mode="bilinear", align_corners=False)

        img = self.net_local(img)

        x = self.compression(torch.cat([x, img], dim=1))

        if self.mode == "encoder":
            x = self.resulotionchange(x)
        if self.mode == "decoder":
            x = upsample(x)

        return x


@MODELS.register_module()
class FSLNet(BaseBackbone):
    def __init__(self,
                 num_img=1,
                 model="L",
                 out_indices=None,
                 norm_eval=False,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(type='Constant', val=1., layer=['_BatchNorm']),
                     dict(type='Normal', std=0.02, layer=['Linear'])
                 ]):
        super(FSLNet, self).__init__(init_cfg)
        self.norm_eval = norm_eval

        if out_indices is None:
            out_indices = (3, )
        self.out_indices = out_indices

        if model == "L":
            self.channels = np.array([32, 64, 128, 256]) * 2
        else:
            self.channels = np.array([32, 64, 128, 256])

        self.encoder = nn.ModuleList()
        self.encoder.append(
            FLBlock(3 * num_img * 2, self.channels[0], 128, 209, mode="encoder")
        )  # [64, 208] *
        self.encoder.append(
            FLBlock(self.channels[0], self.channels[1], 64, 105, mode="encoder")
        )  # [32, 104] *
        self.encoder.append(
            FLBlock(self.channels[1], self.channels[2], 32, 53, mode="encoder")
        )  # [16, 52] *
        self.encoder.append(
            FLBlock(self.channels[2], self.channels[3], 16, 27, mode="encoder")
        )  # [8, 26] *

    def forward(self, x):
        features = []

        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self.out_indices:
                features.append(x)

        return tuple(features)

    def train(self, mode=True):
        super(FSLNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()