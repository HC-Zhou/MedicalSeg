# -*- coding:utf-8  -*-
"""
Time: 2022/10/27 14:07
Author: Yimohanser
Software: PyCharm
"""
import math

from torch import nn
from collections import OrderedDict
from lib.sseg.GLNet.SSA import shunted_t, Block as STB
from lib.sseg.HST.HST_UNet import Block as MSCA


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.act1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                          kernel_size=3, padding=1, stride=2,
                                          output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels // 2)
        self.act2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels // 2, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.deconv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        return x


class LGF(nn.Module):
    def __init__(self, dim, num_heads: int = 4, mlp_ratio: int = 4):
        super(LGF, self).__init__()
        self.Local = MSCA(dim)
        self.Global = STB(dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim), nn.ReLU()
        )

    def forward(self, x):
        B = x.shape[0]
        H = W = int(math.sqrt(x.shape[1]))
        Global = self.Global(x, H, W)
        Global = Global.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        Local = self.Local(x)
        out = Local + Global
        out = self.conv(out)
        return out


class GLNet(nn.Module):
    def __init__(self, img_size=256, pretrained='', num_classes=1):
        super(GLNet, self).__init__()
        self.encoder = shunted_t(img_size=img_size, pretrained=pretrained)
        self.decoder4 = UpSampleBlock(in_channels=512, out_channels=256)
        self.decoder3 = UpSampleBlock(in_channels=256, out_channels=128)
        self.decoder2 = UpSampleBlock(in_channels=128, out_channels=64)
        self.decoder1 = UpSampleBlock(in_channels=64, out_channels=32)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 3, padding=1, bias=False),
        )

    def forward(self, x):
        result = OrderedDict()
        e1, e2, e3, e4 = self.encoder(x)
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.final(d1)
        result['out'] = out
        return result


if __name__ == "__main__":
    from torchsummary import summary

    model = GLNet(img_size=256)
    summary(model, input_size=(3, 256, 256), device='cpu')
