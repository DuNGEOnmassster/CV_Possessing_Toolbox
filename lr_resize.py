import torch
import os
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2

image_path = "./test_dataset/"
target_path = "./target_dataset/lr_resize/"


def get_paired_lrsr():
    for root , dirs, files in os.walk(image_path):
        for name in files:
            print("\n" + name)





if __name__ == "__main__":
    get_paired_lrsr()