import os
from PIL import Image
import numpy as np
 
image_path = "./test_dataset/"
target_path = "./target_dataset/"

 
def cut_image(image, cut_w=128, cut_h=128):
    width, height = image.size
    w_cut_num = width // cut_w
    h_cut_num = height // cut_h

    print(f"w cut {w_cut_num}, h cut {h_cut_num}")
    # box_list = []
    # for i in range(0,3):
    #     for j in range(0,3):
    #         #print((i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width))
    #         box = (j*item_width,i*item_width,(j+1)*item_width,(i+1)*item_width)
    #         box_list.append(box)
    
    # image_list = [image.crop(box) for box in box_list]

    # return image_list


for root , dirs, files in os.walk(image_path):
    for name in files:
        print("\n" + name)
        image = Image.open(root + name)
        # cut_image(image)