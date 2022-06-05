import torch
import argparse
import pickle
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms as T
from typing import Union

from easyface.recognition.models import *
from easyface.utils.visualize import draw_box_and_landmark, show_image
from easyface.utils.io import WebcamStream
from detect_align import FaceDetectAlign


class Inference:
    def __init__(self, face_data:str, model: str, checkpoint: str, det_model: str, det_checkpoint: str, recog_threshold: float) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.recog_threshold = recog_threshold

        face_data = pickle.load(open(face_data, 'rb'))
        self.face_embeds = face_data['embeddings']
        self.labels = face_data['ids']

        self.model = eval(model)(112)
        self.model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.align = FaceDetectAlign(det_model, det_checkpoint)

        self.preprocess = T.Compose([
            T.Lambda(lambda x: x / 255),
            T.Normalize([0.5], [0.5]),
        ])

    def cosine_similarity(self, feats: np.ndarray):
        similarity_scores = feats @ self.face_embeds.T
        print(similarity_scores)
        scores = np.max(similarity_scores, axis=1)
        inds = np.argsort(similarity_scores, axis=1)[:, -1]
        return scores, inds

    def visualize(self, image, dets, scores, inds):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        boxes, landmarks = dets[:, :4].astype(int), dets[:, 5:].astype(int)

        for box, score, ind, landmark in zip(boxes, scores, inds, landmarks):
            if score < self.recog_threshold: 
                text = f"Unknown: {int(score*100):2d}%"
                color = (0, 255, 0)
            else:
                text = f"{self.labels[ind]}: {int(score*100):3d}%"
                color = (0, 0, 255)
            draw_box_and_landmark(image, box, text, landmark, color)
        return image
        
    def __call__(self, img_path: Union[str, np.ndarray]):
        faces, dets, image = self.align.detect_and_align_faces(img_path, (112, 112))
        pfaces = self.preprocess(faces.permute(0, 3, 1, 2)).to(self.device)
        
        with torch.inference_mode():
            feats = self.model(pfaces).detach().cpu().numpy()

        scores, inds = self.cosine_similarity(feats)
        image = self.visualize(image[0], dets[0], scores, inds)
        return image
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='assets/rdj_tom.jpeg')
    parser.add_argument('--face_data', type=str, default='assets/faces.pkl')
    parser.add_argument('--model', type=str, default='AdaFace')
    parser.add_argument('--checkpoint', type=str, default='/home/sithu/checkpoints/FR/adaface/adaface_ir18_webface4m.pth')
    parser.add_argument('--det_model', type=str, default='RetinaFace')
    parser.add_argument('--det_checkpoint', type=str, default='/home/sithu/checkpoints/FR/retinaface/mobilenet0.25_Final.pth')
    parser.add_argument('--recog_threshold', type=float, default=0.3)
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
