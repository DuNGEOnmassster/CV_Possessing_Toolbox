import torch
import os
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F


image_path = "./test_dataset/"
target_path = "./target_dataset/lr_resize/"

LR_path = "xxxx_LR/"
HR_path = "xxxx_HR/"


def get_HRname_from_LRname(LR_name):
    return LR_name.replace("LR", "HR")


def get_paired_lrhr():
    paired_lrhr = []
    for root , dirs, files in os.walk(image_path + LR_path):
        for name in files:
            if name.endswith(".png"):       
                print ("Find LR image: " + os.path.join(root, name))
                HR_name = get_HRname_from_LRname(name)
                lr_img = Image.open(root + name)
                hr_img = Image.open(image_path + HR_path + HR_name)
                lr_img = lr_img.resize((hr_img.size[0], hr_img.size[1]), Image.Resampling.BICUBIC)
                # print("After BICUBIC:")
                print(f"lr size = {lr_img.size}, hr size = {hr_img.size}")
                paired_lrhr.append([lr_img, hr_img])
                toTensor = transforms.ToTensor()
                print(toTensor(lr_img).view(1,-1,lr_img.size[1],lr_img.size[0]).shape)

    return paired_lrhr
    

def save_paired(paired_lrhr):
    index = 1
    for item in paired_lrhr:
        item[0].save(target_path + str(index) + '_LR.png', 'PNG')
        item[1].save(target_path + str(index) + '_HR.png', 'PNG')
        index += 1

if __name__ == "__main__":
    paired_lrhr = get_paired_lrhr()
    # save_paired(paired_lrhr)