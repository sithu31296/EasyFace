import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder, folder


class CustomImageFolder(ImageFolder):
    def __init__(self, 
            root: str, 
            transform=None, 
            target_transform=None, 
            loader=folder.default_loader, 
            is_valid_file=None
        ):
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        self.root = root


    def __getitem__(self, index: int):
        img_path, target = self.samples[index]
        sample = self.loader(img_path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def augment(self, sample):
        