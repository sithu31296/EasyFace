import argparse
import numbers
import cv2
from pathlib import Path
from tqdm import tqdm
from PIL import Image

try:
    import mxnet as mx
except:
    print("Please install `mxnet` for converting from MXRecord file to Images.")



def save_images(rec_path):
    save_path = rec_path / 'imgs'
    save_path.mkdir(exist_ok=True)

    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path / 'train.idx'), str(rec_path / 'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])

    for idx in tqdm(range(1, max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)

        if not isinstance(header.label, numbers.Number):
            label = int(header.label[0])
        else:
            label = int(header.label)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img)
        label_path = save_path / str(label)
        label_path.mkdir(exist_ok=True)

        img.save(label_path / f"{idx}.jpg", quality=95)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rec_path', type=str, default='/home/sithu/datasets/faces_emore')
    args = parser.parse_args()
    rec_path = Path(args.rec_path)
    save_images(rec_path)