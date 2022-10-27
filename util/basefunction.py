# -*- coding:utf-8  -*-
"""
Time: 2022/8/10 14:53
Author: Yimohanser
Software: PyCharm
"""
import glob
import json
import os
import torch
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold
from torchvision.transforms import transforms
from tqdm import tqdm


def mean_std(img, w=256, h=256):
    count = 0
    R = 0
    R_channel_square = 0
    G = 0
    G_channel_square = 0
    B = 0
    B_channel_square = 0
    t = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor()
    ])
    for i, m in enumerate(tqdm(img)):
        m = Image.open(m)
        m = t(m)
        count += w * h
        R += torch.sum(m[0, :, :])
        R_channel_square += torch.sum(torch.pow(m[0, :, :], 2.0))
        G += torch.sum(m[1, :, :])
        G_channel_square += torch.sum(torch.pow(m[1, :, :], 2.0))
        B += torch.sum(m[2, :, :])
        B_channel_square += torch.sum(torch.pow(m[2, :, :], 2.0))

    R_mean = R / count
    G_mean = G / count
    B_mean = B / count

    R_std = np.sqrt(R_channel_square / count - R_mean * R_mean)
    G_std = np.sqrt(G_channel_square / count - G_mean * G_mean)
    B_std = np.sqrt(B_channel_square / count - B_mean * B_mean)

    return [round(R_mean.item(), 3), round(G_mean.item(), 3), round(B_mean.item(), 3)], \
           [round(R_std.item(), 3), round(G_std.item(), 3), round(B_std.item(), 3)]


def read_root(root):
    return glob.glob(os.path.join(root, '*.jpg'))


def dataset_kfold(dataset_dir, k=5):
    indexes = [l[:-4] for l in os.listdir(dataset_dir)]

    kf = KFold(k, shuffle=True)  # k折交叉验证

    val_index = dict()
    train_index = dict()
    for i in range(k):
        val_index[str(i)] = []
        train_index[str(i)] = []

    for i, (train, val) in enumerate(kf.split(indexes)):
        for item in val:
            val_index[str(i)].append(indexes[item])
        for item in train:
            train_index[str(i)].append(indexes[item])

        print('fold:{},train_len:{},val_len:{}'.format(i, len(train), len(val)))

    with open('val.json', 'w') as f:
        json.dump(val_index, f)

    with open('train.json', 'w') as f:
        json.dump(train_index, f)


if __name__ == '__main__':
    # ([0.708, 0.582, 0.536], [0.156, 0.165, 0.18])
    # imgs = read_root(root=r'D:\迅雷下载\ISIC2016\ISBI2016_ISIC_Part1_Training_Data')
    # print(mean_std(imgs, w=512, h=512))
    dataset_kfold(dataset_dir=r'D:\迅雷下载\ISIC2018\train\img', k=5)

    # root = 'D:\迅雷下载\PH2Dataset\PH2 Dataset images'
    # for chdir in os.listdir(root):
    #     img = os.path.join(root, chdir, f'{chdir}_Dermoscopic_Image', f'{chdir}.bmp')
    #     mask = os.path.join(root, chdir, f'{chdir}_lesion', f'{chdir}_lesion.bmp')
