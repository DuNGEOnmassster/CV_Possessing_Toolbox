import os
import albumentations
import numpy as np
import torch.nn as nn
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import utils as vutils
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Image data loader")

    parser.add_argument('--dataset_path', type=str, default='/Users/normanz/Desktop/Github/CV_Possessing_Toolbox/test_dataset/',
                        help='Declare the dataset path')
    parser.add_argument('--LR_path', type=str, default='xxxx_LR',
                        help='Declare LR path in dataset')
    parser.add_argument('--HR_path', type=str, default='xxxx_HR',
                        help='Declare the dataset path')
    parser.add_argument("--save_path", type=str, default="/Users/normanz/Desktop/Github/CV_Possessing_Toolbox/target_dataset/paired_sr",
                        help="Declare save path to check whether lr&hr images are paired")
    parser.add_argument('--device', type=str, default="cuda",
                        help='Declare whether to use cuda')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Declare batch size for training')

    return parser.parse_args()

class ImagePaths(Dataset):
    def __init__(self, path, size=None):
        self.size = size

        self.images = [os.path.join(path, file) for file in os.listdir(path)]
        self._length = len(self.images)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example


def load_data(args):
    HR_data = ImagePaths(args.dataset_path + args.HR_path, size=256)
    LR_data = ImagePaths(args.dataset_path + args.LR_path, size=256)
    print(f"len HR data: {len(HR_data)}, len LR data: {len(LR_data)}")
    train_loader = DataLoader(HR_data+LR_data, batch_size=args.batch_size, shuffle=False)
    return train_loader

if __name__ == "__main__":
    args = parse_args()
    train_dataset = load_data(args)
    i = 1
    for data in train_dataset:
        print(data.shape)
        vutils.save_image(data, os.path.join(args.save_path, f"{i}.jpg"))
        i += 1