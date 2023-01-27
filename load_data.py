import torch
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="dataset params")

    parser.add_argument("--local_dataset", type=bool, default=True,
                        help="declare dataset from local or from site")
    parser.add_argument("--dataset_path", type=str, default='/Users/normanz/Desktop/Github/CV_Possessing_Toolbox/test_dataset/xxxx_HR',
                        help="declare local dataset path")
    parser.add_argument("--dataset_link", type=str, required=False,
                        help="declare site dataset link")
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Declare batch size for training')

    return parser.parse_args()


def get_dataloader(args):
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    # Get Dataset
    if not args.local_dataset:
        os.system(f"wget -P {args.dataset_path} {args.dataset_link}")

    train_dataset = torchvision.datasets.ImageFolder(args.dataset_path, train=True, transform=data_transform["train"])
    test_dataset = torchvision.datasets.ImageFolder(args.dataset_path, train=False, transform=data_transform["val"])
    
    print(f"length of trainning dataset: {len(train_dataset)}")
    print(f"length of testing dataset: {len(test_dataset)}")
    # Get DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    return train_dataloader,test_dataloader

if __name__ == "__main__":
    args = parse_args()
    train_dataloader,test_dataloader = get_dataloader(args)