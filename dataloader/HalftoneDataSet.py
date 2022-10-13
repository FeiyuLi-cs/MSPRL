import os
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import glob

from dataloader.transforms import *


def get_data(halftone_path, target_path):
    halftone_path = glob.glob(halftone_path)
    target_path = glob.glob(target_path)
    return halftone_path, target_path


class PairDataSet(data.Dataset):
    def __init__(self, mode, halftone_path, target_path, image_size, image_channel):
        self.mode = mode
        self.halftone_path, self.target_path = get_data(halftone_path, target_path)
        self.image_channel = image_channel
        self.train_transform = PairCompose(
            [
                PairRandomCrop(image_size),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])
        self.length = len(self.target_path)

    def __len__(self):
        return len(self.target_path)

    def __getitem__(self, index):
        if self.mode == 'train':

            target_path = self.target_path[index]
            target_img = Image.open(target_path)

            halftone_path = self.halftone_path[index]
            halftone_img = Image.open(halftone_path)

            halftone_img, target_img = self.train_transform(halftone_img, target_img)

            return halftone_img, target_img
        elif self.mode == 'valid':

            target_path = self.target_path[index]
            target_img = Image.open(target_path)
            target_img = self.val_transform(target_img)

            halftone_path = self.halftone_path[index]
            halftone_img = Image.open(halftone_path)
            halftone_img = self.val_transform(halftone_img)

            return halftone_img, target_img
