import torch
import argparse
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms as T
from typing import Union

from easyface.emotion.models import *
from easyface.utils.visualize import draw_box_and_landmark, show_image
from easyface.utils.io import WebcamStream
from detect_align import FaceDetectAlign


class Inference:
    def __init__(self, model: str, dataset: str, checkpoint: str, det_model: str, det_checkpoint: str) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']
        assert dataset in ['AffectNet7', 'AffectNet8', 'RAFDB']

        if dataset == 'AffectNet8':
            self.labels.append('Contempt')
        elif dataset == 'RAFDB':
            self.labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

        self.model = eval(model)(len(self.labels))
        self.model.load_checkpoint(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.align = FaceDetectAlign(det_model, det_checkpoint)

        self.preprocess = T.Compose([
            T.Lambda(lambda x: x / 255),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def visualize(self, image, dets, labels, scores):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        boxes, landmarks = dets[:, :4].astype(int), dets[:, 5:].astype(int)

        for box, score, label, landmark in zip(boxes, scores, labels, landmarks):
            text = f"{label}: {int(score*100):2d}%"
            draw_box_and_landmark(image, box, text, landmark, draw_lmks=False)
        return image

    def postprocess(self, preds: torch.Tensor):
        preds = preds.softmax(dim=1)
        probs, idxs = torch.max(preds, dim=1)
        return [self.labels[idx] for idx in idxs], probs.tolist()
        
    def __call__(self, img_path: Union[str, np.ndarray]):
        faces, dets, image = self.align.detect_and_crop_faces(img_path, (224, 224))
        if len(faces) > 0:
            pfaces = self.preprocess(faces.permute(0, 3, 1, 2)).to(self.device)
            
            with torch.inference_mode():
                preds = self.model(pfaces)[0].detach().cpu()
            labels, scores = self.postprocess(preds)

        image = self.visualize(image[0], dets[0], labels, scores)
        return image
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='assets/test.jpg')
    parser.add_argument('--dataset', type=str, default='AffectNet8')
    parser.add_argument('--model', type=str, default='DAN')
    parser.add_argument('--checkpoint', type=str, default='/home/sithu/checkpoints/face_emotion/affecnet8_epoch5_acc0.6209.pth')
    parser.add_argument('--det_model', type=str, default='RetinaFace')
    parser.add_argument('--det_checkpoint', type=str, default='/home/sithu/checkpoints/FR/retinaface/mobilenet0.25_Final.pth')
    args = vars(parser.parse_args())
    source = args.pop('source')
    file_path = Path(source)

    inference = Inference(**args)

    if file_path.is_file():
        image = inference(str(file_path))
        image = Image.fromarray(image[:, :, ::-1]).convert('RGB')
        image.show()
    
    elif str(file_path) == 'webcam':
        stream = WebcamStream(int(str(file_path)))

        for frame in stream:
            frame = inference(frame)
            
            if not show_image(frame):
                break
        stream.stop()

    else:
        raise FileNotFoundError
