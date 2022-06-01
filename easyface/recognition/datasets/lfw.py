import torch
import os
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision import io



class LFW(Dataset):
    """Labeled Faces in the Wild Dataset
    http://vis-www.cs.umass.edu/lfw/

    """
    def __init__(self, root, annot_file, mode='train', transform=None) -> None:
        super().__init__()
        self.mode = mode

        with open(annot_file) as f:
            images = f.read().splitlines()

        self.images = [os.path.join(root, img) for img in images]
        random.shuffle(self.images)

        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)
        

    def __getitem__(self, index: int):
        sample = self.images[index]
        img_path, label = sample.split()
        image = io.read_image(img_path, io.ImageReadMode.GRAY)

        if self.transform is not None:
            image = self.transform(image)

        return image, label.long()



if __name__ == '__main__':
    from torchvision.utils import make_grid
    mode = 'test'
    if mode == 'train':
        transform = T.Compose([
            T.RandomCrop((128, 128)),
            T.RandomHorizontalFlip(),
            T.Lambda(lambda x: x / 255),
            T.Normalize([0.5], [0.5])
        ])
    else:
        transform = T.Compose([
            T.CenterCrop((128, 128)),
            T.Lambda(lambda x: x / 255),
            T.Normalize([0.5], [0.5])
        ])
    dataset = LFW('data/Datasets/fv/dataset_v1.1/dataset_mix_aligned_v1.1', 'data/Datasets/fv/dataset_v1.1/mix_20w.txt', 'test', transform)
    dataloader = DataLoader(dataset, 4)

    for i, (img, label) in enumerate(dataloader):
        img = make_grid(img)
        img = torch.permute(img, [1, 2, 0])
        