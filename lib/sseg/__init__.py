# -*- coding:utf-8  -*-
"""
Time: 2022/8/3 19:34
Author: Yimohanser
Software: PyCharm
"""
from .HST import *
from .baseline import BaseLine
from .GLNet import GLNet

def SSegmentationSet(model: str, num_classes=5, pretrained='', img_size=512):
    if model == 'hst_unet':
        return HST_UNet(img_size=img_size, pretrained=pretrained, num_classes=num_classes)
    elif model == 'baseline':
        return BaseLine(img_size=img_size, pretrained=pretrained, num_classes=num_classes)
    elif model == 'glnet':
        return GLNet(img_size=img_size, pretrained=pretrained, num_classes=num_classes)