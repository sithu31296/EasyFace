import numpy as np
from pathlib import Path
from base import FaceDataset


class LFW(FaceDataset):
    """Labeled Faces in the Wild Dataset
    http://vis-www.cs.umass.edu/lfw/
    """
    def __init__(self, root: str, mode: str = 'train', transform=None) -> None:
        super().__init__(root, mode, transform)
        identities = self.read_split_file(root, mode)
        self.reduce_to_sample_identities(identities)
        self.num_classes = len(np.unique(self.ids))
        print(f" [{mode.capitalize()}] Number of Identities: {self.num_classes}\tNumber of Images: {len(self.ids)}")

    def read_split_file(self, root: str, mode: str = 'train'):
        root = Path(root)
        text_file = 'peopleDevTrain.txt' if mode == 'train' else 'peopleDevTest.txt'
        split_file = root / text_file
        assert split_file.exists()
        identities = []

        with open(split_file) as f:
            lines = f.read().splitlines()[1:]

        for line in lines:
            identity = line.split()[0]
            identities.append(identity)
        return identities



if __name__ == '__main__':
    dataset = LFW('/home/sithu/datasets/lfw-align-128', 'test')
    dataset.visualize_dataset_samples(dataset, 4)