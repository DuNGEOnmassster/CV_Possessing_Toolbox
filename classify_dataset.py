import torch
import os
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F


def get_HRname_from_LRname(LR_name):
    return LR_name.replace("LR", "HR")


def get_paired_lrhr(dataset_path):
    paired_lrhr = []
    for root , dirs, files in os.walk(dataset_path):
        for name in files:
            if name.endswith("LR.png"):       
                print ("Find LR image: " + os.path.join(root, name))
                HR_name = get_HRname_from_LRname(name)
                lr_img = Image.open(os.path.join(root, name))
                hr_img = Image.open(os.path.join(root, HR_name))
                lr_img = lr_img.resize((hr_img.size[0], hr_img.size[1]), Image.Resampling.BICUBIC)
                # print("After BICUBIC:")
                print(f"lr size = {lr_img.size}, hr size = {hr_img.size}")
                paired_lrhr.append([lr_img, hr_img])
                toTensor = transforms.ToTensor()
                # print(toTensor(lr_img).view(1,-1,lr_img.size[1],lr_img.size[0]).shape)

    return paired_lrhr


if __name__ == "__main__":
    dataset_path = "./target_dataset/"
    # LR_path = "xxxx_LR/"
    # HR_path = "xxxx_HR/"

    paired_lrhr = get_paired_lrhr(dataset_path)
    # print(paired_lrhr)