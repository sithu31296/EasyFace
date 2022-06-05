import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import io
from torchvision.utils import make_grid


class FaceDataset(Dataset):
    def __init__(self, root: str, mode: str = 'train', transform = None) -> None:
        super().__init__()
        self.transform = transform
        self.img_paths, self.labels, self.ids = self.scan_root(root)
        # self.reduce_to_num_identities(10)

        self.num_classes = len(np.unique(self.ids))
        print(f" [Total] Number of Identities: {self.num_classes}\tNumber of Images: {len(self.ids)}")

    def scan_root(self, root: str):
        root = Path(root)
        dirs = sorted([dir for dir in root.iterdir() if dir.is_dir()], key=lambda x: x.stem.lower())
        img_paths, labels, ids = [], [], []

        for id, dir in enumerate(dirs):
            if dir.is_dir():
                for img_path in dir.glob('*.jpg'):
                    img_paths.append(img_path)
                    labels.append(dir.stem)
                    ids.append(id)
        return img_paths, labels, ids

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index):
        img = io.read_image(str(self.img_paths[index]))
        label = self.ids[index]

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def reduce_to_num_identities(self, n: int) -> None:
        unique_ids = np.unique(self.ids)
        assert n > 0 and n <= len(unique_ids)
        chosen_ids = unique_ids[:n]
        chosen_idxs = np.where(np.isin(self.ids, chosen_ids))[0]
        self.img_paths = [self.img_paths[idx] for idx in chosen_idxs]
        self.ids = [self.ids[idx] for idx in chosen_idxs]
        self.labels = [self.labels[idx] for idx in chosen_idxs]

    def reduce_to_sample_identities(self, identities: list):
        chosen_idxs = np.where(np.isin(self.labels, identities))[0]
        chosen_ids = np.unique(np.array(self.ids)[chosen_idxs])
        label_map = {id: n for n, id in enumerate(chosen_ids)}

        self.img_paths = [self.img_paths[idx] for idx in chosen_idxs]
        self.ids = [label_map[self.ids[idx]] for idx in chosen_idxs]
        self.labels = [self.labels[idx] for idx in chosen_idxs]
        
    @staticmethod
    def visualize_dataset_samples(dataset: Dataset, num_samples: int) -> None:
        dataloader = DataLoader(dataset, num_samples, shuffle=True)
        images, label = next(iter(dataloader))
        print(f" Image Shape\t: {images.shape[1:]}")
        print(f" Identities\t: {label.tolist()}")
        print(f" Names\t: {[dataset.labels[lbl] for lbl in label.tolist()]}")

        img_grid = make_grid(images, num_samples // 2).to(torch.uint8).numpy().transpose(1, 2, 0)
        plt.imshow(img_grid)
        plt.show()