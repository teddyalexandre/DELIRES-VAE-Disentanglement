from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image
import numpy as np
from torch.utils.data import random_split


class dSpritesDataset(Dataset):
    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)
        return img

class RescaleBinaryImage:
    def __call__(self, tensor):
        return tensor / torch.max(tensor)


def get_dataloaders(dataset_path, batch_size = 64) :

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load dataset
    dataset_zip = np.load(dataset_path, encoding='bytes')
    imgs = dataset_zip['imgs']

    dsprites = dSpritesDataset(imgs, transform=transform)

    subset_size = 15000
    dsprites_small = random_split(dsprites, [subset_size, len(imgs)-subset_size])[0]
    imgs_train, imgs_test = random_split(dsprites_small, [0.8, 0.2])

    # Build dataset
    train_dataset = dSpritesDataset(imgs_train, transform = transform)
    test_dataset = dSpritesDataset(imgs_test, transform = transform)

    # Build dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    return train_dataloader, test_dataloader