import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="dataset params")

    return parser.parse_args()


def get_dataloader(batch_size, args):
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
    train_dataset = torchvision.datasets.ImageFolder(args.data_dir, train=True, transform=data_transform["train"], download=False)
    test_dataset = torchvision.datasets.ImageFolder(args.data_dir, train=False, transform=data_transform["val"], download=False)
    print('length of trainning dataset: {}'.format(len(train_dataset)))
    print('length of testing dataset: {}'.format(len(test_dataset)))
    # Get DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader,test_dataloader

