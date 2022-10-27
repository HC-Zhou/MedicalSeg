# -*- coding:utf-8  -*-
"""
Time: 2022/10/18 17:18
Author: Yimohanser
Software: PyCharm
"""
import collections
import json
import os

from torch.utils import data
from PIL import Image
from torchvision import transforms
from ..utils.ext_transforms import ExtendTransforms


class ISIC_Dataset(data.Dataset):
    def __init__(self, root, mean, std,
                 img_size=512, mode="train",
                 fold=0):
        super(ISIC_Dataset, self).__init__()
        self.mode = mode
        self.root = root
        self.mean = mean
        self.std = std
        self.img_size = img_size

        self.files = collections.defaultdict(list)
        for split_file in ['train', 'val']:
            json_file = json.load(open(self.root + f'/{split_file}.json', 'r'))
            for img_name in json_file[f'{fold}']:
                img_file = self.root + f'/img/{img_name}.jpg'
                lbl_file = self.root + f'/mask/{img_name}_segmentation.png'
                self.files[split_file].append({
                    'img': img_file,
                    'mask': lbl_file,
                })

        self.x_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        self.y_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size),
                              interpolation=transforms.InterpolationMode.BILINEAR),
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

    def __len__(self):
        return len(self.files[self.mode])


class ISIC2016(data.Dataset):
    def __init__(self, root, mean, std, img_size=512):
        super(ISIC2016, self).__init__()
        self.root = root
        self.mean = mean
        self.std = std
        self.img_size = img_size

        self.files = collections.defaultdict(list)
        for img_name in os.listdir(os.path.join(root, 'ISBI2016_ISIC_Part1_Training_Data')):
            img_name = img_name[:-4]
            img_file = self.root + f'/ISBI2016_ISIC_Part1_Training_Data/{img_name}.jpg'
            lbl_file = self.root + f'/ISBI2016_ISIC_Part1_Training_GroundTruth/{img_name}_Segmentation.png'
            self.files['train'].append({
                'img': img_file,
                'mask': lbl_file,
            })

        self.x_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        self.y_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        # Read Data
        item = self.files['train'][idx]
        image_path = item['img']
        mask_path = item['mask']

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # Data Augmentation
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

    def __len__(self):
        return len(self.files['train'])


class PH2(data.Dataset):
    def __init__(self, root, mean, std, img_size=512):
        super(PH2, self).__init__()
        self.root = root
        self.mean = mean
        self.std = std
        self.img_size = img_size

        self.files = collections.defaultdict(list)
        self.names = os.listdir(self.root)

        for img_name in self.names:
            img_file = self.root + f'/{img_name}/{img_name}_Dermoscopic_Image/{img_name}.bmp'
            lbl_file = self.root + f'/{img_name}/{img_name}_lesion/{img_name}_lesion.bmp'
            self.files['train'].append({
                'img': img_file,
                'mask': lbl_file,
            })

        self.x_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        self.y_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        # Read Data
        item = self.files['train'][idx]
        image_path = item['img']
        mask_path = item['mask']

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # Resize image
        img = self.x_transform(image)
        mask = self.y_transform(mask)
        ori = self.img_transform(image)

        return img, mask, ori

    def __len__(self):
        return len(self.files['train'])
