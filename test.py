import os
import argparse
import albumentations
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import utils as vutils

# --------------------------------------------- #
#                  Data Utils
# --------------------------------------------- #

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


class ImagePaths_pairs(Dataset):
    def __init__(self, path_hr, path_lr, size_hr=None, size_lr=None):
        self.size_hr = size_hr
        self.size_lr = size_lr

        self.images_hr = [os.path.join(path_hr, file) for file in os.listdir(path_hr)]
        self.images_lr = [os.path.join(path_lr, file) for file in os.listdir(path_lr)]
        self._length = len(self.images_hr)
        # 默认两个数据集等长

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size_hr)
        self.cropper = albumentations.CenterCrop(height=self.size_hr, width=self.size_hr)
        self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        
        self.rescaler_lr = albumentations.SmallestMaxSize(max_size=self.size_lr)
        self.cropper_lr = albumentations.CenterCrop(height=self.size_lr, width=self.size_lr)
        self.preprocessor_lr = albumentations.Compose([self.rescaler_lr, self.cropper_lr])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path, for_hr = True):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        if for_hr:
            image = self.preprocessor(image=image)["image"]
        else:
            image = self.preprocessor_lr(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example_hr = self.preprocess_image(self.images_hr[i])
        example_lr = self.preprocess_image(self.images_lr[i], for_hr=False)
        example = [example_hr, example_lr]
        # print(example)
        return example


def load_data(args):
    train_data = ImagePaths(args.dataset_path, size=args.image_size) # original size = 256
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    return train_loader

def load_data_pairs(args):
    train_data_paires = ImagePaths_pairs(args.dataset_path, args.dataset_path_lr, size_hr=args.image_size, size_lr=args.image_size_lr)
    train_loader_pairs = DataLoader(train_data_paires, batch_size=args.batch_size, shuffle=True)
    return train_loader_pairs


# --------------------------------------------- #
#                  Module Utils
#            for Encoder, Decoder etc.
# --------------------------------------------- #

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(images):
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    full_sample = images["full_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].imshow(full_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    # The hyperparameters used in VQGAN_SR are listed here...
    parser = argparse.ArgumentParser(description="VQGAN_SR")
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--image-size-lr', type=int, default=64, help='Image height and width (default: 64)')
    parser.add_argument('--size-z', type=int, default=64, help='Latent size of code z')
    parser.add_argument('--channel-z', type=int, default=64, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--rrdb-nb', type=int, default=3, help='Number of RRDB block')
    parser.add_argument('--rrdb-middle-feature-channel', type=int, default=64, help='the channel dimension of middle variables in RRDB, == channel_z')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--dataset-path-lr', type=str, default='/dataLR', help='Path to dataLR (default: /dataLR)')
    parser.add_argument('--device', type=str, default="cuda:6", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=8, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
    parser.add_argument('--disc_start', type=int, default=1000, help='When to start the discriminator (default: 0)')
    parser.add_argument('--disc_factor', type=float, default=1., help='')
    parser.add_argument('--rec_loss_factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual_loss_factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument("--save_path", type=str, default="/Users/normanz/Desktop/Github/CV_Possessing_Toolbox/target_dataset/paired_sr",
                        help="Declare save path to check whether lr&hr images are paired")
    args = parser.parse_args()
    args.dataset_path = "/Users/normanz/Desktop/Github/CV_Possessing_Toolbox/test_dataset/xxxx_HR"
    args.dataset_path_lr = "/Users/normanz/Desktop/Github/CV_Possessing_Toolbox/test_dataset/xxxx_LR"
    
    train_dataset_pairs = load_data_pairs(args)
    
    for i, imgs in enumerate(train_dataset_pairs):
        print(imgs[0].shape)
        print(imgs[1].shape)
        vutils.save_image(imgs[0], os.path.join(args.save_path, f"{i}_HR.jpg"))
        vutils.save_image(imgs[1], os.path.join(args.save_path, f"{i}_LR.jpg"))
