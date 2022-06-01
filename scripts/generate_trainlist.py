import argparse
from pathlib import Path


def main(root, output):
    root = Path(root)
    folders = list(root.glob('*'))
    print(f"Number of Identities: {len(folders)}")
    file_lists = []
    label = 0
    
    for folder in folders:
        if folder.is_dir():
            img_paths = folder.glob('*')
            for img_path in img_paths:
                full_path = root / img_path
                file_lists.append(f"{str(full_path)} {label}\n")
            label += 1

    print(f"Number of Images: {len(file_lists)}")
    
    with open(output, 'w') as f:
        f.writelines(file_lists)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/sithu/datasets/lfw_split/train')
    parser.add_argument('--output', type=str, default='lfw_train.txt')
    args = parser.parse_args()

    main(**vars(args))