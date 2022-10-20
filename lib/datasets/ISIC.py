# -*- coding:utf-8  -*-
"""
Time: 2022/10/18 17:18
Author: Yimohanser
Software: PyCharm
"""
from PIL import Image
from torchvision import transforms
from .base_dataset import SingleClassBaseDataset
from ..utils.ext_transforms import ExtendTransforms


class ISIC_Dataset(SingleClassBaseDataset):
    def __init__(self, root,
                 mean, std,
                 img_size=512, mode="train",
                 fold=0):
        super(ISIC_Dataset, self).__init__(root=root, mean=mean, std=std,
                                           img_size=img_size, mode=mode, fold=fold)
        self.x_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        self.y_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        # Read Data
        item = self.files[self.mode][idx]
        image_path = item['img']
        mask_path = item['mask']

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # Data Augmentation
        if self.mode == 'train':
            image, mask = ExtendTransforms.flip(image, mask)
            image, mask = ExtendTransforms.rotate(image, mask)
            image, mask = ExtendTransforms.randomResizeCrop(image, mask)
            image, mask = ExtendTransforms.adjust_hue(image, mask)
            image, mask = ExtendTransforms.adjustContrast(image, mask)
            image, mask = ExtendTransforms.adjustSaturation(image, mask)

        # Resize image
        img = self.x_transform(image)
        mask = self.y_transform(mask)
        ori = self.img_transform(image)

        return img, mask, ori
