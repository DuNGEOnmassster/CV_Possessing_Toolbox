import os
import sys
from PIL import Image
import numpy as np
 
image_path = "./test_dataset/"
target_path = "./target_dataset/mean_cut/"

 
def cut_image(image, cut_w=128, cut_h=128):
    width, height = image.size
    w_cut_num = width // cut_w
    h_cut_num = height // cut_h

    print(f"w cut {w_cut_num}, h cut {h_cut_num}")
    box_list = []
    for i in range(0, w_cut_num):
        for j in range(0, h_cut_num):
            box = (j*cut_h,i*cut_w,(j+1)*cut_h,(i+1)*cut_w)
            box_list.append(box)
    
    image_list = [image.crop(box) for box in box_list]

    return image_list


def save_images(image_list, name):
    target_dir = target_path + name.split(sep=".png")[0] + "/"
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    index = 1
    for image in image_list:
        image.save(target_dir + str(index) + '.png', 'PNG')
        index += 1


def possess():
    for root , dirs, files in os.walk(image_path):
        for name in files:
            print("\n" + name)
            image = Image.open(root + name)
            image_list = cut_image(image)
            save_images(image_list, name)

possess()