import os
import argparse
import albumentations
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import utils as vutils
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.nn.functional as F
import torchvision
import random

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


# --------------------------------------------- #
#                  Data Utils
# --------------------------------------------- #
class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        # self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
 
    def forward(self, input, target, feature_layers=[0,1,2,3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        # input = (input-self.mean) / self.std
        # target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.mse_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.mse_loss(gram_x, gram_y)
        return loss

class ImagePaths(Dataset):
    def __init__(self, path, size=None):
        self.size = size

        self.images = [os.path.join(path, file) for file in os.listdir(path)]
        self._length = len(self.images)

        # self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.cropper])

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
        self.t = size_hr//size_lr

        self.images_hr = [os.path.join(path_hr, file) for file in os.listdir(path_hr)]
        self.images_lr = [os.path.join(path_lr, file) for file in os.listdir(path_lr)]
        self._length = len(self.images_hr)
        # 默认两个数据集等长

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path_hr, image_path_lr):
        image_hr = Image.open(image_path_hr)
        if not image_hr.mode == "RGB":
            image_hr = image_hr.convert("RGB")
        image_hr = np.array(image_hr).astype(np.uint8)
        
        image_lr = Image.open(image_path_lr)
        if not image_lr.mode == "RGB":
            image_lr = image_lr.convert("RGB")
        image_lr = np.array(image_lr).astype(np.uint8)

        height, width = image_lr.shape[:2]
        x_min = random.randint(0, width-self.size_lr) if width-self.size_lr > 0 else 0
        y_min = random.randint(0, height-self.size_lr) if height-self.size_lr > 0 else 0
        # hr
        preprocessor_hr = albumentations.Crop(x_min*self.t, y_min*self.t,
                                              x_min*self.t + self.size_hr, y_min*self.t + self.size_hr)
        image_hr = preprocessor_hr(image=image_hr)["image"]
        image_hr = (image_hr / 127.5 - 1.0).astype(np.float32)
        image_hr = image_hr.transpose(2, 0, 1)
        # lr
        preprocessor_lr = albumentations.Crop(x_min, y_min, x_min + self.size_lr, y_min + self.size_lr)
        image_lr = preprocessor_lr(image=image_lr)["image"]
        image_lr = (image_lr / 127.5 - 1.0).astype(np.float32)
        image_lr = image_lr.transpose(2, 0, 1)
        return image_hr, image_lr

    def __getitem__(self, i):
        example_hr, example_lr = self.preprocess_image(self.images_hr[i], self.images_lr[i])
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

    args = parser.parse_args()
    args.dataset_path = "./test_dataset/xxxx_HR"
    args.dataset_path_lr = "./test_dataset/xxxx_LR"
    
    train_dataset_pairs = load_data_pairs(args)
    os.makedirs("pair_results", exist_ok=True)
    
    for i, imgs in enumerate(train_dataset_pairs):
        print(imgs[0].shape)
        print(imgs[1].shape)
        vutils.save_image(imgs[0], os.path.join("pair_results", f"hr_{i}.jpg"), nrow=4)
        vutils.save_image(imgs[1], os.path.join("pair_results", f"lr_{i}.jpg"), nrow=4)
        print(f"saved: {i}")

