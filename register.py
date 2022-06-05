import argparse
import torch
import pickle
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
            T.Normalize([0.5], [0.5])
        ])

    def __call__(self, img_paths: str):
        identities = [img_path.stem for img_path in img_paths]
        faces = self.align.detect_and_align_faces(img_paths, (112, 112))[0]
        pfaces = self.preprocess(faces.permute(0, 3, 1, 2)).to(self.device)
        
        with torch.inference_mode():
            features = self.model(pfaces)
        return features.detach().cpu().numpy(), identities
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='assets/test_faces')
    parser.add_argument('--output', type=str, default='assets/faces.pkl')
    parser.add_argument('--model', type=str, default='AdaFace')
    parser.add_argument('--checkpoint', type=str, default='/home/sithu/checkpoints/FR/adaface/adaface_ir18_webface4m.pth')
    parser.add_argument('--det_model', type=str, default='RetinaFace')
    parser.add_argument('--det_checkpoint', type=str, default='/home/sithu/checkpoints/FR/retinaface/mobilenet0.25_Final.pth')
    args = vars(parser.parse_args())
    source = args.pop('source')
    output = args.pop('output')
    file_path = Path(source)
    output_path = Path(output).parent
    output_path.mkdir(exist_ok=True)

    inference = Inference(**args)

    if file_path.is_dir():
        image_paths = list(file_path.glob('*'))
        features, identities = inference(image_paths)
        pickle.dump({'embeddings': features, 'ids': identities}, open(output, 'wb'))
    
    else:
        raise FileNotFoundError