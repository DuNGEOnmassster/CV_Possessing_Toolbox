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
    parser.add_argument("--dataset_path", type=str, default='/Users/normanz/Desktop/Github/CV_Possessing_Toolbox/test_dataset',
                        help="declare local dataset path")
    parser.add_argument("--dataset_link", type=str, required=False,
                        help="declare site dataset link")
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Declare batch size for training')
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="declare proportion of train split in dataset")
    parser.add_argument("--valid_split", type=float, default=0.1,
                        help="declare proportion of valid split in dataset")
    parser.add_argument("--test_split", type=float, default=0.1,
                        help="declare proportion of test split in dataset")
    parser.add_argument("--random_seed", type=int, default=3407,
                        help="use this magical random seed 3407")

    return parser.parse_args()


def transform_label(x):
    return torch.tensor([x]).float()


def get_loader(args):
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

    origin_dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=data_transform["train"], target_transform=transform_label)
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(origin_dataset,[int(len(origin_dataset)*args.train_split), int(len(origin_dataset)*args.valid_split), int(len(origin_dataset)*args.test_split)])
    
    print(f"length of trainning dataset: {len(train_dataset)}")
    print(f"length of validing dataset: {len(valid_dataset)}")
    print(f"length of testing dataset: {len(test_dataset)}")
    # Get DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    args = parse_args()
    train_loader, valid_loader, test_loader = get_loader(args)
    for target, label in train_loader:
        print(target.shape, label.shape)