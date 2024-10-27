# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer

from mmpretrain.registry import MODELS
from .resnet import ResNet, get_expansion
from .vision_attention import (
    SpatialAttention, ChannelAttention,  # CBAM
    SpatialGate, ChannelGate,  # BAM

)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


@MODELS.register_module()
class ResNet_Freq(ResNet):
    """Freq ResNet backbone for RSI forgery detection.

    Compared to standard ResNet, 该模型首先将原始图像转为频率域,
     然后再进行特征提取.

    Args:
        depth (int): Network depth, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Output channels of the stem layer. Default: 64.
        base_channels (int): Middle channels of the first stage. Default: 64.
        num_stages (int): Stages of the network. Default: 4.
    """

    def __init__(self, depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 deep_stem=False, **kwargs):
        super(ResNet_Freq, self).__init__(
            depth, deep_stem=deep_stem, **kwargs)
        assert not self.deep_stem, 'ResNet_Freq do not support deep_stem'

        self.conv1 = nn.Conv2d(in_channels, stem_channels, kernel_size=1, stride=1, bias=True)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, stem_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)

        self.realconv1 = conv1x1(stem_channels, stem_channels, stride=1)
        self.imagconv1 = conv1x1(stem_channels, stem_channels, stride=1)

        self.conv2 = nn.Conv2d(stem_channels, stem_channels, kernel_size=1, stride=2, bias=True)
        self.realconv2 = conv1x1(stem_channels, stem_channels, stride=1)
        self.imagconv2 = conv1x1(stem_channels, stem_channels, stride=1)

        _channels = base_channels*self.expansion
        self.conv3 = nn.Conv2d(_channels, _channels, kernel_size=1, stride=1, bias=True)
        self.realconv3 = conv1x1(_channels, _channels, stride=1)
        self.imagconv3 = conv1x1(_channels, _channels, stride=1)

        self.conv4 = nn.Conv2d(_channels, _channels, kernel_size=1, stride=1, bias=True)
        self.realconv4 = conv1x1(_channels, _channels, stride=1)
        self.imagconv4 = conv1x1(_channels, _channels, stride=1)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _make_stem_layer(self, in_channels, base_channels):
        # self.conv1 = build_conv_layer(
        #     self.conv_cfg,
        #     in_channels,
        #     base_channels,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        #     bias=False)
        # self.norm1_name, norm1 = build_norm_layer(
        #     self.norm_cfg, base_channels, postfix=1)
        # self.add_module(self.norm1_name, norm1)
        # self.relu = nn.ReLU(inplace=True)
        pass

    def hfreqWH(self, x, scale):  # 高频表示模块
        assert scale > 2
        # print(f'input shape: {x.shape}, min: {x.min()}, max: {x.max()}, mean: {x.mean()}')
        x = torch.fft.fft2(
            x, norm="ortho"
        )  # ,norm='forward'  #使用 torch.fft.fft2 对图像进行快速傅里叶变换，将图像转换到频域。
        x = torch.fft.fftshift(
            x, dim=[-2, -1]
        )  # 使用 torch.fft.fftshift 对频谱进行中心化处理，将零频率分量移到频谱中心。
        b, c, h, w = x.shape  # 通过设置中心区域（大小为scale）的频谱分量为0，
        x[
            :,
            :,
            h // 2 - h // scale:h // 2 + h // scale,
            w // 2 - w // scale:w // 2 + w // scale,
        ] = 0.0  # 实现高通滤波，移除低频信息，保留高频信息。
        x = torch.fft.ifftshift(
            x, dim=[-2, -1]
        )  # 使用 torch.fft.ifftshift 对频谱进行反中心化处理。
        x = torch.fft.ifft2(
            x, norm="ortho"
        )  # 使用 torch.fft.ifft2 对频谱进行逆快速傅里叶变换，将图像转换回空间域。
        x = torch.real(x)  # 取实部作为高频表示结果。
        x = F.relu(x, inplace=True)  # 通过 torch.relu 进行激活
        # print(f'output shape: {x.shape}, min: {x.min()}, max: {x.max()}, mean: {x.mean()}')
        # print()
        return x

    def hfreqC(self, x, scale):
        assert scale > 2
        # print(f'input shape: {x.shape}, min: {x.min()}, max: {x.max()}, mean: {x.mean()}')
        x = torch.fft.fft(x, dim=1, norm="ortho")  # ,norm='forward'
        x = torch.fft.fftshift(x, dim=1)
        b, c, h, w = x.shape

        x[:, c // 2 - c // scale:c // 2 + c // scale, :, :] = 0.0

        x = torch.fft.ifftshift(x, dim=1)
        x = torch.fft.ifft(x, dim=1, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)

        # print(f'output shape: {x.shape}, min: {x.min()}, max: {x.max()}, mean: {x.mean()}')
        # print()
        return x

    def forward(self, x):

        # *** Covn1 ***
        ### HFRI  高频表示模块
        x = self.hfreqWH(x, 4)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        ### HFRFC
        x = self.hfreqC(x, 4)

        ### FCL
        x = torch.fft.fft2(x, norm="ortho")  # ,norm='forward'
        x = torch.fft.fftshift(x, dim=[-2, -1])
        x = torch.complex(self.realconv1(x.real), self.imagconv1(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = self.relu(x)

        ### HFRFS
        x = self.hfreqWH(x, 4)
        x = self.conv2(x)
        x = self.relu(x)

        ### HFRFC
        x = self.hfreqC(x, 4)

        ### FCL
        x = torch.fft.fft2(x, norm="ortho")  # ,norm='forward'
        x = torch.fft.fftshift(x, dim=[-2, -1])
        x = torch.complex(self.realconv2(x.real), self.imagconv2(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = self.relu(x)

        x = self.maxpool(x)  # size/2

        # *** Layer1 ***
        outs = []
        x = self.layer1(x)  # in64 out256

        ### HFRFS
        x = self.hfreqWH(x, 4)
        x = self.conv3(x)
        x = self.relu(x)

        ### FCL
        x = torch.fft.fft2(x, norm="ortho")  # ,norm='forward'
        x = torch.fft.fftshift(x, dim=[-2, -1])
        x = torch.complex(self.realconv3(x.real), self.imagconv3(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = self.relu(x)

        ### HFRFS
        x = self.hfreqWH(x, 4)
        x = self.conv4(x)
        x = self.relu(x)

        ### FCL
        x = torch.fft.fft2(x, norm="ortho")  # ,norm='forward'
        x = torch.fft.fftshift(x, dim=[-2, -1])
        x = torch.complex(self.realconv4(x.real), self.imagconv4(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = self.relu(x)

        outs.append(x)  # Save output of layer1

        # *** Layer2 ***
        x = self.layer2(x)
        outs.append(x)
        # *** Layer3 ***
        x = self.layer3(x)
        outs.append(x)
        # *** Layer4 ***
        x = self.layer4(x)
        outs.append(x)

        # for i, layer_name in enumerate(self.res_layers):
        #     res_layer = getattr(self, layer_name)
        #     x = res_layer(x)
        #     if i in self.out_indices:
        #         outs.append(x)
        return tuple(outs)


@MODELS.register_module()
class ResNet_Freq_Dual_v1(ResNet):
    """Freq ResNet backbone for RSI forgery detection.

    Compared to standard ResNet, 该模型采用频率域和图像域双通路.
    """

    def __init__(self, depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 deep_stem=False, drop_path_rate=0.0,
                 **kwargs):
        super(ResNet_Freq_Dual_v1, self).__init__(
            depth, in_channels=in_channels, stem_channels=stem_channels,
            base_channels=base_channels, deep_stem=deep_stem, drop_path_rate=drop_path_rate, **kwargs)

        self.conv1_freq = nn.Conv2d(in_channels, stem_channels, kernel_size=1, stride=1, bias=True)
        _, self.norm1_freq = build_norm_layer(self.norm_cfg, stem_channels, postfix='1_freq')

        self.realconv1 = conv1x1(stem_channels, stem_channels, stride=1)
        self.imagconv1 = conv1x1(stem_channels, stem_channels, stride=1)

        self.conv2_freq = nn.Conv2d(stem_channels, stem_channels, kernel_size=1, stride=2, bias=True)
        self.realconv2 = conv1x1(stem_channels, stem_channels, stride=1)
        self.imagconv2 = conv1x1(stem_channels, stem_channels, stride=1)

        _in_channels = stem_channels
        _out_channels = base_channels * self.expansion

        # stochastic depth decay rule
        total_depth = sum(self.stage_blocks)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        for i, num_blocks in enumerate(self.stage_blocks[:1]):
            stride = self.strides[i]
            dilation = self.dilations[i]
            # Build layer1 for freq domain
            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=num_blocks,
                in_channels=_in_channels,
                out_channels=_out_channels,
                expansion=self.expansion,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=self.with_cp,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                drop_path_rate=dpr[:num_blocks])
            layer_name = f'layer{i + 1}_freq'
            self.add_module(layer_name, res_layer)

            _in_channels = _out_channels
            _out_channels *= 2
            dpr = dpr[num_blocks:]
        print('Build Backbone of ResNet_Freq_Dual_v1!!!')

    # def _make_stem_layer(self, in_channels, base_channels):
    #     pass

    def hfreqWH(self, x, scale):  # 高频表示模块
        assert scale > 2
        # print(f'input shape: {x.shape}, min: {x.min()}, max: {x.max()}, mean: {x.mean()}')
        x = torch.fft.fft2(
            x, norm="ortho"
        )  # ,norm='forward'  #使用 torch.fft.fft2 对图像进行快速傅里叶变换，将图像转换到频域。
        x = torch.fft.fftshift(
            x, dim=[-2, -1]
        )  # 使用 torch.fft.fftshift 对频谱进行中心化处理，将零频率分量移到频谱中心。
        b, c, h, w = x.shape  # 通过设置中心区域（大小为scale）的频谱分量为0，
        x[
            :,
            :,
            h // 2 - h // scale:h // 2 + h // scale,
            w // 2 - w // scale:w // 2 + w // scale,
        ] = 0.0  # 实现高通滤波，移除低频信息，保留高频信息。
        x = torch.fft.ifftshift(
            x, dim=[-2, -1]
        )  # 使用 torch.fft.ifftshift 对频谱进行反中心化处理。
        x = torch.fft.ifft2(
            x, norm="ortho"
        )  # 使用 torch.fft.ifft2 对频谱进行逆快速傅里叶变换，将图像转换回空间域。
        x = torch.real(x)  # 取实部作为高频表示结果。
        x = F.relu(x, inplace=True)  # 通过 torch.relu 进行激活
        # print(f'output shape: {x.shape}, min: {x.min()}, max: {x.max()}, mean: {x.mean()}')
        # print()
        return x

    def hfreqC(self, x, scale):
        assert scale > 2
        # print(f'input shape: {x.shape}, min: {x.min()}, max: {x.max()}, mean: {x.mean()}')
        x = torch.fft.fft(x, dim=1, norm="ortho")  # ,norm='forward'
        x = torch.fft.fftshift(x, dim=1)
        b, c, h, w = x.shape

        x[:, c // 2 - c // scale:c // 2 + c // scale, :, :] = 0.0

        x = torch.fft.ifftshift(x, dim=1)
        x = torch.fft.ifft(x, dim=1, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)

        # print(f'output shape: {x.shape}, min: {x.min()}, max: {x.max()}, mean: {x.mean()}')
        # print()
        return x

    def forward(self, x):

        # *** Covn1 ***
        x_feat = self.conv1(x)  # size/2
        x_feat = self.norm1(x_feat)
        x_feat = self.relu(x_feat)
        x_feat = self.maxpool(x_feat)  # size/4

        ### HFRI  高频表示模块
        x_freq = self.hfreqWH(x, 4)
        x_freq = self.conv1_freq(x_freq)
        x_freq = self.norm1_freq(x_freq)
        x_freq = self.relu(x_freq)

        ### HFRFC
        x_freq = self.hfreqC(x_freq, 4)

        ### FCL
        x_freq = torch.fft.fft2(x_freq, norm="ortho")  # ,norm='forward'
        x_freq = torch.fft.fftshift(x_freq, dim=[-2, -1])
        x_freq = torch.complex(self.realconv1(x_freq.real), self.imagconv1(x_freq.imag))
        x_freq = torch.fft.ifftshift(x_freq, dim=[-2, -1])
        x_freq = torch.fft.ifft2(x_freq, norm="ortho")
        x_freq = torch.real(x_freq)
        x_freq = self.relu(x_freq)

        ### HFRFS
        x_freq = self.hfreqWH(x_freq, 4)
        x_freq = self.conv2_freq(x_freq)  # size/2
        x_freq = self.relu(x_freq)

        ### HFRFC
        x_freq = self.hfreqC(x_freq, 4)

        ### FCL
        x_freq = torch.fft.fft2(x_freq, norm="ortho")  # ,norm='forward'
        x_freq = torch.fft.fftshift(x_freq, dim=[-2, -1])
        x_freq = torch.complex(self.realconv2(x_freq.real), self.imagconv2(x_freq.imag))
        x_freq = torch.fft.ifftshift(x_freq, dim=[-2, -1])
        x_freq = torch.fft.ifft2(x_freq, norm="ortho")
        x_freq = torch.real(x_freq)
        x_freq = self.relu(x_freq)

        x_freq = self.maxpool(x_freq)  # size/4

        # *** Layer1 ***
        outs = []
        x_feat = self.layer1(x_feat)  # in64 out256
        outs.append(x_feat)  # Save output of layer1

        x_freq = self.layer1_freq(x_freq)  # in64 out256

        x = x_feat + x_freq

        # *** Layer2 ***
        x = self.layer2(x)
        outs.append(x)
        # *** Layer3 ***
        x = self.layer3(x)
        outs.append(x)
        # *** Layer4 ***
        x = self.layer4(x)
        outs.append(x)

        return tuple(outs)


@MODELS.register_module()
class ResNet_Freq_Dual_CBAM_simple(ResNet_Freq_Dual_v1):
    """Freq ResNet backbone for RSI forgery detection.

    Compared to standard ResNet, 该模型采用频率域和图像域双通路.
    并引入了Attention机制(CBAM).
    """

    def __init__(self, depth, base_channels=64, **kwargs):
        super(ResNet_Freq_Dual_CBAM_simple, self).__init__(
            depth, base_channels=base_channels, **kwargs)

        # Branch Attention (after layer1)
        layer1_out_channels = base_channels * self.expansion
        self.branch_attention = nn.Sequential(
            ChannelAttention(layer1_out_channels*2, ratio=16),
            nn.Conv2d(layer1_out_channels*2, layer1_out_channels, 1, bias=False),
            SpatialAttention(kernel_size=7),
        )
        print('Build Backbone of ResNet_Freq_Dual_CBAM_simple!!!')

    def forward(self, x):

        # *** Covn1 ***
        x_feat = self.conv1(x)
        x_feat = self.norm1(x_feat)
        x_feat = self.relu(x_feat)
        x_feat = self.maxpool(x_feat)  # size/2

        ### HFRI  高频表示模块
        x_freq = self.hfreqWH(x, 4)
        x_freq = self.conv1_freq(x_freq)
        x_freq = self.norm1_freq(x_freq)
        x_freq = self.relu(x_freq)

        ### HFRFC
        x_freq = self.hfreqC(x_freq, 4)

        ### FCL
        x_freq = torch.fft.fft2(x_freq, norm="ortho")  # ,norm='forward'
        x_freq = torch.fft.fftshift(x_freq, dim=[-2, -1])
        x_freq = torch.complex(self.realconv1(x_freq.real), self.imagconv1(x_freq.imag))
        x_freq = torch.fft.ifftshift(x_freq, dim=[-2, -1])
        x_freq = torch.fft.ifft2(x_freq, norm="ortho")
        x_freq = torch.real(x_freq)
        x_freq = self.relu(x_freq)

        ### HFRFS
        x_freq = self.hfreqWH(x_freq, 4)
        x_freq = self.conv2_freq(x_freq)
        x_freq = self.relu(x_freq)

        ### HFRFC
        x_freq = self.hfreqC(x_freq, 4)

        ### FCL
        x_freq = torch.fft.fft2(x_freq, norm="ortho")  # ,norm='forward'
        x_freq = torch.fft.fftshift(x_freq, dim=[-2, -1])
        x_freq = torch.complex(self.realconv2(x_freq.real), self.imagconv2(x_freq.imag))
        x_freq = torch.fft.ifftshift(x_freq, dim=[-2, -1])
        x_freq = torch.fft.ifft2(x_freq, norm="ortho")
        x_freq = torch.real(x_freq)
        x_freq = self.relu(x_freq)

        x_freq = self.maxpool(x_freq)  # size/2

        # *** Layer1 ***
        outs = []
        x_feat = self.layer1(x_feat)  # in64 out256
        if 0 in self.out_indices:
            outs.append(x_feat)  # Save output of layer1

        x_freq = self.layer1_freq(x_freq)  # in64 out256

        x = torch.cat([x_feat, x_freq], dim=1)
        x = self.branch_attention(x)

        # *** Layer2 ***
        x = self.layer2(x)
        if 1 in self.out_indices:
            outs.append(x)
        # *** Layer3 ***
        x = self.layer3(x)
        if 2 in self.out_indices:
            outs.append(x)
        # *** Layer4 ***
        x = self.layer4(x)
        if 3 in self.out_indices:
            outs.append(x)

        return tuple(outs)


@MODELS.register_module()
class ResNet_Freq_Dual_CBAM_simple_v2(ResNet_Freq_Dual_v1):
    """Freq ResNet backbone for RSI forgery detection.

    Compared to standard ResNet, 该模型采用频率域和图像域双通路.
    并引入了Attention机制(CBAM).
    """

    def __init__(self, depth, base_channels=64, **kwargs):
        super(ResNet_Freq_Dual_CBAM_simple_v2, self).__init__(
            depth, base_channels=base_channels, **kwargs)

        # Branch Attention (after layer1)
        layer1_out_channels = base_channels * self.expansion
        self.feat_attention = nn.Sequential(
            ChannelAttention(layer1_out_channels, ratio=8),
            SpatialAttention(kernel_size=7),
        )
        self.freq_attention = nn.Sequential(
            ChannelAttention(layer1_out_channels, ratio=8),
            SpatialAttention(kernel_size=7),
        )
        print('Build Backbone of ResNet_Freq_Dual_CBAM_simple!!!')

    def forward(self, x):

        # *** Covn1 ***
        x_feat = self.conv1(x)
        x_feat = self.norm1(x_feat)
        x_feat = self.relu(x_feat)
        x_feat = self.maxpool(x_feat)  # size/2

        ### HFRI  高频表示模块
        x_freq = self.hfreqWH(x, 4)
        x_freq = self.conv1_freq(x_freq)
        x_freq = self.norm1_freq(x_freq)
        x_freq = self.relu(x_freq)

        ### HFRFC
        x_freq = self.hfreqC(x_freq, 4)

        ### FCL
        x_freq = torch.fft.fft2(x_freq, norm="ortho")  # ,norm='forward'
        x_freq = torch.fft.fftshift(x_freq, dim=[-2, -1])
        x_freq = torch.complex(self.realconv1(x_freq.real), self.imagconv1(x_freq.imag))
        x_freq = torch.fft.ifftshift(x_freq, dim=[-2, -1])
        x_freq = torch.fft.ifft2(x_freq, norm="ortho")
        x_freq = torch.real(x_freq)
        x_freq = self.relu(x_freq)

        ### HFRFS
        x_freq = self.hfreqWH(x_freq, 4)
        x_freq = self.conv2_freq(x_freq)
        x_freq = self.relu(x_freq)

        ### HFRFC
        x_freq = self.hfreqC(x_freq, 4)

        ### FCL
        x_freq = torch.fft.fft2(x_freq, norm="ortho")  # ,norm='forward'
        x_freq = torch.fft.fftshift(x_freq, dim=[-2, -1])
        x_freq = torch.complex(self.realconv2(x_freq.real), self.imagconv2(x_freq.imag))
        x_freq = torch.fft.ifftshift(x_freq, dim=[-2, -1])
        x_freq = torch.fft.ifft2(x_freq, norm="ortho")
        x_freq = torch.real(x_freq)
        x_freq = self.relu(x_freq)

        x_freq = self.maxpool(x_freq)  # size/2

        # *** Layer1 ***
        outs = []
        x_feat = self.layer1(x_feat)  # in64 out256
        if 0 in self.out_indices:
            outs.append(x_feat)  # Save output of layer1

        x_freq = self.layer1_freq(x_freq)  # in64 out256

        x_feat = self.feat_attention(x_feat)
        x_freq = self.freq_attention(x_freq)
        x = x_feat + x_freq

        # *** Layer2 ***
        x = self.layer2(x)
        if 1 in self.out_indices:
            outs.append(x)
        # *** Layer3 ***
        x = self.layer3(x)
        if 2 in self.out_indices:
            outs.append(x)
        # *** Layer4 ***
        x = self.layer4(x)
        if 3 in self.out_indices:
            outs.append(x)

        return tuple(outs)


@MODELS.register_module()
class ResNet_Freq_Dual_CBAM(ResNet_Freq_Dual_v1):
    """Freq ResNet backbone for RSI forgery detection.

    Compared to standard ResNet, 该模型采用频率域和图像域双通路.
    并引入了Attention机制(CBAM).
    """

    def __init__(self, depth,
                 base_channels=64,
                 **kwargs):
        super(ResNet_Freq_Dual_CBAM, self).__init__(
            depth, base_channels=base_channels, **kwargs)

        self.conv1_sa = SpatialAttention(kernel_size=7)
        # Branch Attention (after layer1)
        layer1_out_channels = base_channels * self.expansion
        self.branch_attention = nn.Sequential(
            ChannelAttention(layer1_out_channels*2, ratio=16),
            nn.Conv2d(layer1_out_channels*2, layer1_out_channels, 1, bias=False)
        )

        # Spatial Attention (layer1-4)
        self.layer1_sa = SpatialAttention(kernel_size=7)
        self.layer2_sa = SpatialAttention(kernel_size=7)
        self.layer3_sa = SpatialAttention(kernel_size=7)
        self.layer4_sa = SpatialAttention(kernel_size=7)
        print('Build Backbone of ResNet_Freq_Dual_CBAM!!!')

    def forward(self, x):

        # *** Covn1 ***
        x_feat = self.conv1(x)
        x_feat = self.norm1(x_feat)
        x_feat = self.conv1_sa(x_feat)  # Spatial Attention
        x_feat = self.relu(x_feat)
        x_feat = self.maxpool(x_feat)  # size/2

        ### HFRI  高频表示模块
        x_freq = self.hfreqWH(x, 4)
        x_freq = self.conv1_freq(x_freq)
        x_freq = self.norm1_freq(x_freq)
        x_freq = self.relu(x_freq)

        ### HFRFC
        x_freq = self.hfreqC(x_freq, 4)

        ### FCL
        x_freq = torch.fft.fft2(x_freq, norm="ortho")  # ,norm='forward'
        x_freq = torch.fft.fftshift(x_freq, dim=[-2, -1])
        x_freq = torch.complex(self.realconv1(x_freq.real), self.imagconv1(x_freq.imag))
        x_freq = torch.fft.ifftshift(x_freq, dim=[-2, -1])
        x_freq = torch.fft.ifft2(x_freq, norm="ortho")
        x_freq = torch.real(x_freq)
        x_freq = self.relu(x_freq)

        ### HFRFS
        x_freq = self.hfreqWH(x_freq, 4)
        x_freq = self.conv2_freq(x_freq)
        x_freq = self.relu(x_freq)

        ### HFRFC
        x_freq = self.hfreqC(x_freq, 4)

        ### FCL
        x_freq = torch.fft.fft2(x_freq, norm="ortho")  # ,norm='forward'
        x_freq = torch.fft.fftshift(x_freq, dim=[-2, -1])
        x_freq = torch.complex(self.realconv2(x_freq.real), self.imagconv2(x_freq.imag))
        x_freq = torch.fft.ifftshift(x_freq, dim=[-2, -1])
        x_freq = torch.fft.ifft2(x_freq, norm="ortho")
        x_freq = torch.real(x_freq)
        x_freq = self.relu(x_freq)

        x_freq = self.maxpool(x_freq)  # size/2

        # *** Layer1 ***
        outs = []
        x_feat = self.layer1(x_feat)  # in64 out256
        x_feat = self.layer1_sa(x_feat)  # Spatial Attention
        if 0 in self.out_indices:
            outs.append(x)  # Save output of layer1

        x_freq = self.layer1_freq(x_freq)  # in64 out256

        x = torch.cat([x_feat, x_freq], dim=1)
        x = self.branch_attention(x)

        # *** Layer2 ***
        x = self.layer2(x)
        x = self.layer2_sa(x)  # Spatial Attention
        if 1 in self.out_indices:
            outs.append(x)

        # *** Layer3 ***
        x = self.layer3(x)
        x = self.layer3_sa(x)  # Spatial Attention
        if 2 in self.out_indices:
            outs.append(x)

        # *** Layer4 ***
        x = self.layer4(x)
        x = self.layer4_sa(x)  # Spatial Attention
        if 3 in self.out_indices:
            outs.append(x)

        return tuple(outs)


@MODELS.register_module()
class ResNet_Freq_Dual_CBAM_v2(ResNet_Freq_Dual_v1):
    """Freq ResNet backbone for RSI forgery detection.

    Compared to standard ResNet, 该模型采用频率域和图像域双通路.
    并引入了Attention机制(CBAM).
    """

    def __init__(self, depth,
                 stem_channels=64,
                 base_channels=64,
                 **kwargs):
        super(ResNet_Freq_Dual_CBAM_v2, self).__init__(
            depth, stem_channels=stem_channels, base_channels=base_channels, **kwargs)

        self.freq_ca = ChannelAttention(stem_channels, ratio=16)

        _out_channels = base_channels * self.expansion
        for i, _ in enumerate(self.stage_blocks):
            # Build layer(1) for freq domain
            if i == 0:
                # Branch Attention (after layer1)
                self.branch_attention = nn.Sequential(
                    ChannelAttention(_out_channels*2, 16),
                    nn.Conv2d(_out_channels*2, _out_channels, 1, bias=False)
                )

            # Build BCAM for layer(2-3)
            if i > 0:
                ca_layer = ChannelAttention(_out_channels, 16)
                layer_name = f'layer{i + 1}_ca'
                self.add_module(layer_name, ca_layer)
                sa_layer = SpatialAttention(kernel_size=7)
                layer_name = f'layer{i + 1}_sa'
                self.add_module(layer_name, sa_layer)

            # Update layer settings
            _out_channels *= 2

        print('Build Backbone of ResNet_Freq_Dual_CBAM_v2')

    def forward(self, x):

        # *** Covn1 ***
        x_feat = self.conv1(x)
        x_feat = self.norm1(x_feat)
        x_feat = self.relu(x_feat)
        x_feat = self.maxpool(x_feat)  # size/2

        ### HFRI  高频表示模块
        x_freq = self.hfreqWH(x, 4)
        x_freq = self.conv1_freq(x_freq)
        x_freq = self.norm1_freq(x_freq)
        x_freq = self.relu(x_freq)

        ### HFRFC
        x_freq = self.hfreqC(x_freq, 4)

        ### FCL
        x_freq = torch.fft.fft2(x_freq, norm="ortho")  # ,norm='forward'
        x_freq = torch.fft.fftshift(x_freq, dim=[-2, -1])
        x_freq = torch.complex(self.realconv1(x_freq.real), self.imagconv1(x_freq.imag))
        x_freq = torch.fft.ifftshift(x_freq, dim=[-2, -1])
        x_freq = torch.fft.ifft2(x_freq, norm="ortho")
        x_freq = torch.real(x_freq)
        x_freq = self.freq_ca(x_freq)  # Channel Attention for freq domain
        x_freq = self.relu(x_freq)

        ### HFRFS
        x_freq = self.hfreqWH(x_freq, 4)
        x_freq = self.conv2_freq(x_freq)
        x_freq = self.relu(x_freq)

        ### HFRFC
        x_freq = self.hfreqC(x_freq, 4)

        ### FCL
        x_freq = torch.fft.fft2(x_freq, norm="ortho")  # ,norm='forward'
        x_freq = torch.fft.fftshift(x_freq, dim=[-2, -1])
        x_freq = torch.complex(self.realconv2(x_freq.real), self.imagconv2(x_freq.imag))
        x_freq = torch.fft.ifftshift(x_freq, dim=[-2, -1])
        x_freq = torch.fft.ifft2(x_freq, norm="ortho")
        x_freq = torch.real(x_freq)
        x_freq = self.relu(x_freq)

        x_freq = self.maxpool(x_freq)  # size/2

        # *** Layer1 ***
        outs = []
        x_feat = self.layer1(x_feat)  # in64 out256
        if 0 in self.out_indices:
            outs.append(x_feat)  # Save output of layer1

        x_freq = self.layer1_freq(x_freq)  # in64 out256

        x = torch.cat([x_feat, x_freq], dim=1)
        x = self.branch_attention(x)

        # *** Layer2 ***
        x = self.layer2(x)
        x = self.layer2_sa(self.layer2_ca(x))  # CBAM
        if 1 in self.out_indices:
            outs.append(x)

        # *** Layer3 ***
        x = self.layer3(x)
        x = self.layer3_sa(self.layer3_ca(x))  # CBAM
        if 2 in self.out_indices:
            outs.append(x)

        # *** Layer4 ***
        x = self.layer4(x)
        x = self.layer4_sa(self.layer4_ca(x))  # CBAM
        if 3 in self.out_indices:
            outs.append(x)

        return tuple(outs)


@MODELS.register_module()
class ResNet_Freq_Dual_BAM(ResNet_Freq_Dual_v1):
    """Freq ResNet backbone for RSI forgery detection.

    Compared to standard ResNet, 该模型采用频率域和图像域双通路.
    并引入了Attention机制(BAM).
    """

    def __init__(self, depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 deep_stem=False, drop_path_rate=0.0,
                 **kwargs):
        super(ResNet_Freq_Dual_BAM, self).__init__(
            depth, in_channels=in_channels, stem_channels=stem_channels,
            base_channels=base_channels, deep_stem=deep_stem,
            drop_path_rate=drop_path_rate, **kwargs)

        self.conv1_feat = nn.Conv2d(in_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_sa = SpatialGate(stem_channels, reduction_ratio=16, dilation_conv_num=2, dilation_val=4)

        self.conv1_freq = nn.Conv2d(in_channels, stem_channels, kernel_size=1, stride=1, bias=True)

        self.norm1_freq_name, norm1_freq = build_norm_layer(
            self.norm_cfg, stem_channels, postfix='1_freq')
        self.add_module(self.norm1_freq_name, norm1_freq)
        self.norm1_feat_name, norm1_feat = build_norm_layer(
            self.norm_cfg, stem_channels, postfix='1_feat')
        self.add_module(self.norm1_feat_name, norm1_feat)
        self.relu = nn.ReLU(inplace=True)

        self.realconv1 = conv1x1(stem_channels, stem_channels, stride=1)
        self.imagconv1 = conv1x1(stem_channels, stem_channels, stride=1)

        self.conv2_freq = nn.Conv2d(stem_channels, stem_channels, kernel_size=1, stride=2, bias=True)
        self.realconv2 = conv1x1(stem_channels, stem_channels, stride=1)
        self.imagconv2 = conv1x1(stem_channels, stem_channels, stride=1)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        _in_channels = stem_channels
        _out_channels = base_channels * self.expansion

        # stochastic depth decay rule
        total_depth = sum(self.stage_blocks)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            if i < 1:
                # Build layer1 for freq domain
                res_layer = self.make_res_layer(
                    block=self.block,
                    num_blocks=num_blocks,
                    in_channels=_in_channels,
                    out_channels=_out_channels,
                    expansion=self.expansion,
                    stride=stride,
                    dilation=dilation,
                    style=self.style,
                    avg_down=self.avg_down,
                    with_cp=self.with_cp,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    drop_path_rate=dpr[:num_blocks])
                res_layer_name = f'layer{i + 1}_freq'
                self.add_module(res_layer_name, res_layer)

                # Branch Attention (after layer1)
                self.branch_attention = nn.Sequential(
                    ChannelGate(_out_channels*2),
                    nn.Conv2d(_out_channels*2, _out_channels, 1, bias=False)
                )
            # Cal moudle paramters
            _in_channels = _out_channels
            _out_channels *= 2
            dpr = dpr[num_blocks:]

            # Spatial Attention (layer1-4)
            sa_layer = SpatialGate(_out_channels)
            sa_layer_name = f'layer{i + 1}_sa'
            self.add_module(sa_layer_name, sa_layer)
        print('Build Backbone of ResNet_Freq_Dual_BAM')

    def forward(self, x):

        # *** Covn1 ***
        x_feat = self.conv1_feat(x)
        x_feat = self.bn1_feat(x_feat)
        x_feat = self.relu(x_feat)
        x_feat = self.conv1_sa(x_feat)  # Spatial Attention
        x_feat = self.maxpool(x_feat)  # size/2

        ### HFRI  高频表示模块
        x_freq = self.hfreqWH(x, 4)
        x_freq = self.conv1_freq(x_freq)
        x_freq = self.bn1_freq(x_freq)
        x_freq = self.relu(x_freq)

        ### HFRFC
        x_freq = self.hfreqC(x_freq, 4)

        ### FCL
        x_freq = torch.fft.fft2(x_freq, norm="ortho")  # ,norm='forward'
        x_freq = torch.fft.fftshift(x_freq, dim=[-2, -1])
        x_freq = torch.complex(self.realconv1(x_freq.real), self.imagconv1(x_freq.imag))
        x_freq = torch.fft.ifftshift(x_freq, dim=[-2, -1])
        x_freq = torch.fft.ifft2(x_freq, norm="ortho")
        x_freq = torch.real(x_freq)
        x_freq = self.relu(x_freq)

        ### HFRFS
        x_freq = self.hfreqWH(x_freq, 4)
        x_freq = self.conv2_freq(x_freq)
        x_freq = self.relu(x_freq)

        ### HFRFC
        x_freq = self.hfreqC(x_freq, 4)

        ### FCL
        x_freq = torch.fft.fft2(x_freq, norm="ortho")  # ,norm='forward'
        x_freq = torch.fft.fftshift(x_freq, dim=[-2, -1])
        x_freq = torch.complex(self.realconv2(x_freq.real), self.imagconv2(x_freq.imag))
        x_freq = torch.fft.ifftshift(x_freq, dim=[-2, -1])
        x_freq = torch.fft.ifft2(x_freq, norm="ortho")
        x_freq = torch.real(x_freq)
        x_freq = self.relu(x_freq)

        x_freq = self.maxpool(x_freq)  # size/2

        # *** Layer1 ***
        outs = []
        x_feat = self.layer1(x_feat)  # in64 out256
        x_feat = self.layer1_sa(x_feat)  # Spatial Attention
        outs.append(x_feat)  # Save output of layer1

        x_freq = self.layer1_freq(x_freq)  # in64 out256

        x = torch.cat([x_feat, x_freq], dim=1)
        x = self.branch_attention(x)

        # *** Layer2 ***
        x = self.layer2(x)
        x_feat = self.layer2_sa(x_feat)  # Spatial Attention
        outs.append(x)

        # *** Layer3 ***
        x = self.layer3(x)
        x_feat = self.layer3_sa(x_feat)  # Spatial Attention
        outs.append(x)

        # *** Layer4 ***
        x = self.layer4(x)
        x_feat = self.layer4_sa(x_feat)  # Spatial Attention
        outs.append(x)

        return tuple(outs)
