from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
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


def get_dataloaders(dataset_path, batch_size = 64) :

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load dataset
    dataset_zip = np.load(dataset_path, encoding='bytes')
    imgs = dataset_zip['imgs']
    imgs_train, imgs_test = random_split(imgs)

    # Build dataset
    train_dataset = dSpritesDataset(imgs_train, transform = transform)
    test_dataset = dSpritesDataset(imgs_test, transform = transform)

    # Build dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    return train_dataloader, test_dataloader