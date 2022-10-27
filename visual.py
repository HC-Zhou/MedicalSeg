# -*- coding:utf-8  -*-
"""
Time: 2022/10/27 8:25
Author: Yimohanser
Software: PyCharm
"""
import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from glob import glob
from torchvision import transforms

from lib.sseg import SSegmentationSet


def parse_args():
    parser = argparse.ArgumentParser(description='Visual segmentation network')
    parser.add_argument('--model', default='HST_UNet', type=str)
    parser.add_argument('--img_size', default=512, type=int)
    parser.add_argument('--root', default='./exp/ori', type=str)
    parser.add_argument('--Spath', default='./exp/exp', type=str)
    parser.add_argument('--suffix', default='.png', type=str)
    parser.add_argument('--weight', type=str,
                        default=r'D:\ML\MedicalSeg\saveModels\best_model\ISIC2016.pth')

    args = parser.parse_args()
    return args


def read_root(root, suffix='.png'):
    return glob(os.path.join(root, f'*{suffix}'))


def plot_img_gray(img, output, Spath='.', index="1", img_size=(512, 512)):
    ori_mask = Image.fromarray(output).resize(img_size)

    plt.figure(figsize=(12, 8), dpi=200)

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("ori")

    plt.subplot(1, 2, 2)
    plt.imshow(ori_mask, cmap="gray")
    plt.title("predict")
    plt.savefig(Spath + "//{0}.png".format(index))


def visual():
    args = parse_args()

    model = SSegmentationSet(model=args.model.lower(),
                             num_classes=1,
                             img_size=args.img_size)
    model.load_state_dict(torch.load(args.weight, map_location='cpu')['model'])
    model = model.to('cuda')
    model.eval()

    ori_transform = transforms.Compose([
        transforms.Resize((512, 512),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.724, 0.619, 0.567],
                             std=[0.165, 0.172, 0.193])
    ])

    imgs = read_root(args.root, args.suffix)

    for idx, ori in enumerate(imgs):
        img = Image.open(ori)
        img_tensor = ori_transform(img).unsqueeze(dim=0).to('cuda')
        out = model(img_tensor)['out'].sigmoid().squeeze(1).detach().cpu().numpy()
        out = (out > 0.5).squeeze(0).astype(np.uint8)

        plot_img_gray(img, out, args.Spath, idx, img.size)



if __name__ == "__main__":
    visual()
