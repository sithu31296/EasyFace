import torch
import argparse
from pathlib import Path
from torchvision import transforms as T

from easyface.recognition.models import *
from detect_align import FaceDetectAlign


class Inference:
    def __init__(self, model: str, checkpoint: str, det_model: str, det_checkpoint: str) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = eval(model)(112)
        self.model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.align = FaceDetectAlign(det_model, det_checkpoint)

        self.preprocess = T.Compose([
            T.Lambda(lambda x: x / 255),
            T.Normalize([0.5], [0.5]),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])
        
    def __call__(self, img_path: str):
        face = self.align.detect_and_align_faces(img_path, (112, 112))[0][0]
        pface = self.preprocess(face.permute(2, 0, 1)).to(self.device)
        
        with torch.inference_mode():
            feature = self.model(pface)
        return feature.detach().cpu()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='assets/test_faces')
    parser.add_argument('--model', type=str, default='AdaFace')
    parser.add_argument('--checkpoint', type=str, default='/home/sithu/checkpoints/FR/adaface/adaface_ir18_webface4m.pth')
    parser.add_argument('--det_model', type=str, default='RetinaFace')
    parser.add_argument('--det_checkpoint', type=str, default='/home/sithu/checkpoints/FR/retinaface/mobilenet0.25_Final.pth')
    args = vars(parser.parse_args())
    source = args.pop('source')
    file_path = Path(source)

    inference = Inference(**args)

    if file_path.is_dir():
        image_paths = file_path.glob('*')
        features = []

        for i, image_path in enumerate(image_paths):
            feature = inference(str(image_path))
            features.append(feature)

    similarity_scores = torch.cat(features) @ torch.cat(features).T
    print(similarity_scores)