import torch
import os
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2

image_path = "./test_dataset/"
target_path = "./target_dataset/lr_resize/"

LR_path = "xxxx_LR/"
HR_path = "xxxx_HR/"


def get_HRname_from_LRname(LR_name):
    return LR_name.replace("LR", "HR")


def get_paired_lrsr():
    for root , dirs, files in os.walk(image_path + LR_path):
        for name in files:
            if name.endswith(".png"):       
                print ("Find LR image: " + os.path.join(root, name))
                HR_name = get_HRname_from_LRname(name)
                # print(image_path + HR_path + HR_name)
                lr_img = Image.open(root + name)
                hr_img = Image.open(image_path + HR_path + HR_name)
                print(f"lr size = {lr_img.size}, hr size = {hr_img.size}")
                lr_img = lr_img.resize((hr_img.size[0], hr_img.size[1]), Image.Resampling.BICUBIC)
                print("After BICUBIC:")
                print(f"lr size = {lr_img.size}, hr size = {hr_img.size}")

if __name__ == "__main__":
    get_paired_lrsr()