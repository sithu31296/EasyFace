import argparse
import shutil
from pathlib import Path


def main(root, output):
    root = Path(root)
    output = Path(output)
    train_folder = output / 'train'
    test_folder = output / 'test'

    if output.exists():
        shutil.rmtree(output)
    output.mkdir()
    train_folder.mkdir()
    test_folder.mkdir()

    folders = list(filter(lambda x: x.is_dir(), root.iterdir()))
    train_splits = folders[:int(0.8*len(folders))]
    test_splits = folders[int(0.8*len(folders)):]

    print(f"Total: {len(folders)}\nTrain: {len(train_splits)}\nTest: {len(test_splits)}")

    for folder in train_splits:
        shutil.copytree(folder, train_folder / folder.name)
    for folder in test_splits:
        shutil.copytree(folder, test_folder / folder.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/sithu/datasets/lfw-align-128')
    parser.add_argument('--output', type=str, default='/home/sithu/datasets/lfw_split')
    args = parser.parse_args()

    main(**vars(args))