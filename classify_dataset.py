import torch
import os
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F


def get_HRname_from_LRname(LR_name):
    return LR_name.replace("LR", "HR")


def save_lrhr(lr_imgs, hr_imgs, root):
    # print(f"lr: {lr_imgs}, hr: {hr_imgs}, in root: {root}")
    lr_path = os.path.join(root, "LR")
    hr_path = os.path.join(root, "HR")
    print(lr_path, hr_path)




def get_paired_lrhr(dataset_path):
    paired_lrhr = []
    for root , dirs, files in os.walk(dataset_path):
        # lr_path = os.path.join(root, "LR")
        # hr_path = os.path.join(root, "HR")
        # print(f"lr path: {lr_path}, hr path: {hr_path}")
        save_flag = False
        lr_imgs = []
        hr_imgs = []
        for name in files:
            if name.endswith("LR.png"):   
                save_flag = True    
                print ("Find LR image: " + os.path.join(root, name) + ", in root: " + root)
                HR_name = get_HRname_from_LRname(name)
                lr_img = Image.open(os.path.join(root, name))
                hr_img = Image.open(os.path.join(root, HR_name))
                # lr_img = lr_img.resize((hr_img.size[0], hr_img.size[1]), Image.Resampling.BICUBIC)
                # print("After BICUBIC:")
                # print(f"lr size = {lr_img.size}, hr size = {hr_img.size}")
                # paired_lrhr.append([lr_img, hr_img])
                lr_imgs.append([lr_img, name])
                hr_imgs.append([hr_img, HR_name])
                
                # toTensor = transforms.ToTensor()
                # print(toTensor(lr_img).view(1,-1,lr_img.size[1],lr_img.size[0]).shape)
        if save_flag:
            save_lrhr(lr_imgs, hr_imgs, root)

    return paired_lrhr


if __name__ == "__main__":
    dataset_path = "./target_dataset/"
    # LR_path = "xxxx_LR/"
    # HR_path = "xxxx_HR/"

    paired_lrhr = get_paired_lrhr(dataset_path)
    # print(paired_lrhr)