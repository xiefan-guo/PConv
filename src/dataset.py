import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def check_image_file(filename):
    # ------------------------------------------------------
    # https://www.runoob.com/python/python-func-any.html
    # https://www.runoob.com/python/att-string-endswith.html
    # ------------------------------------------------------
    return any([filename.endswith(extention) for extention in
                ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP']])


def image_transforms(load_size, crop_size):
    # --------------------------------------------------------------
    # https://blog.csdn.net/weixin_38533896/article/details/86028509
    # --------------------------------------------------------------
    return transforms.Compose([
        transforms.Resize(size=load_size, interpolation=Image.BICUBIC),
        transforms.RandomCrop(size=crop_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])


def mask_transforms(crop_size):

    return transforms.Compose([
        transforms.Resize(size=crop_size, interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])


class ImageDataset(Dataset):

    def __init__(self, image_root, mask_root, load_size, crop_size):
        super(ImageDataset, self).__init__()

        self.image_files = [os.path.join(root, file) for root, dirs, files in os.walk(image_root)
                            for file in files if check_image_file(file)]
        self.mask_files = [os.path.join(root, file) for root, dirs, files in os.walk(mask_root)
                           for file in files if check_image_file(file)]

        self.number_image = len(self.image_files)
        self.number_mask = len(self.mask_files)
        self.load_size = load_size
        self.crop_size = crop_size
        self.image_files_transforms = image_transforms(load_size, crop_size)
        self.mask_files_transforms = mask_transforms(crop_size)

    def __getitem__(self, index):

        image = Image.open(self.image_files[index % self.number_image])
        mask = Image.open(self.mask_files[random.randint(0, self.number_mask - 1)])

        ground_truth = self.image_files_transforms(image.convert('RGB'))
        mask = self.mask_files_transforms(mask.convert('RGB'))

        threshold = 0.5
        ones = mask >= threshold
        zeros = mask < threshold

        mask.masked_fill_(ones, 1.0)
        mask.masked_fill_(zeros, 0.0)

        # ---------------------------------------------------
        # white values(ones) denotes the area to be inpainted
        # dark values(zeros) is the values remained
        # ---------------------------------------------------
        mask = 1 - mask
        input_image = ground_truth * mask

        return input_image, ground_truth, mask

    def __len__(self):
        return self.number_image
