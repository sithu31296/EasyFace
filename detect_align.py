import torch
import cv2
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from skimage import transform as trans
from typing import Union, List, Tuple

from easyface.detection.models import *
from easyface.detection.utils.box_utils import PriorBox, decode_boxes, decode_landmarks
from easyface.detection.utils.transform import get_ref_facial_points
from easyface.detection.utils.nms import py_cpu_nms
from easyface.utils.visualize import draw_box_and_landmark


class FaceDetectAlign:
    def __init__(self, model: str, checkpoint: str, conf_threshold: float = 0.6, nms_threshold: float = 0.4) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = eval(model)()
        self.model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.variance = [0.1, 0.2]
        self.mean = np.array((123, 117, 104), dtype=np.float32).reshape(1, 1, 1, 3)
        self.ref_pts = get_ref_facial_points((112, 112))
        self.lmk_transform = trans.SimilarityTransform()
        

    def read_image(self, image: Union[np.ndarray, str, list]) -> np.ndarray:
        if isinstance(image, str):
            image = cv2.imread(image)[:, :, ::-1][np.newaxis, ...]
        elif isinstance(image, list):
            image = [cv2.imread(str(img))[:, :, ::-1] for img in image]
            image = np.stack(image)
        else:
            image = image[np.newaxis, ...]
        return image

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32)
        image -= self.mean
        image = torch.from_numpy(image.transpose(0, 3, 1, 2)).to(self.device)
        return image
    
    def postprocess(self, confidences: torch.Tensor, bboxes: torch.Tensor, landmarks: torch.Tensor, img_size: Tuple[int, int]) -> List[np.ndarray]:
        """
        confidences: classifications, shape: [B, ?, 2]
        bboxes: locations, shape: [B, ?, 4]
        landmarks: landmarks, shape: [B, ?, 10]
        img_size: original image size in [H, W]
        """
        boxes = bboxes.detach().cpu()
        scores = confidences.detach().cpu().numpy()[:, :, 1]
        lmks = landmarks.detach().cpu()

        priorbox = PriorBox(img_size)
        priors = priorbox()

        boxes = decode_boxes(boxes, priors, self.variance, img_size).numpy()
        lmks = decode_landmarks(lmks, priors, self.variance, img_size).numpy()
        
        # do NMS
        dets = []
        for box, score, lmk in zip(boxes, scores, lmks):
            box, score, lmk = py_cpu_nms(box, score, lmk, self.conf_threshold, self.nms_threshold)
            dets.append(np.concatenate([box, score[:, np.newaxis], lmk], axis=1))
        return dets
    
    def detect_and_align_faces(self, img_paths: Union[np.ndarray, str, list], crop_size: Tuple[int, int] = (112, 112)) -> Tuple[torch.Tensor, List[np.ndarray], np.ndarray]:
        all_dets, images = self.detect_faces(img_paths)
        faces = []
        for dets, image in zip(all_dets, images):
            for det in dets:
                warped_face = self.warp_and_crop_face(image, det[5:], crop_size)
                faces.append(torch.as_tensor(warped_face))
        if len(faces) > 0:
            return torch.stack(faces), all_dets, images
        return None, None, images

    def detect_and_crop_faces(self, img_paths: Union[np.ndarray, str, list], crop_size: Tuple[int, int] = (112, 112)) -> Tuple[torch.Tensor, List[np.ndarray], np.ndarray]:
        all_dets, images = self.detect_faces(img_paths)
        faces = []
        for dets, image in zip(all_dets, images):
            boxes = dets[:, :4].astype(int)
            for box in boxes:
                x1, y1, x2, y2 = box
                face = image[y1:y2, x1:x2, :]
                if face.shape[0] <=0 or face.shape[1] <= 0:
                    continue
                face = cv2.resize(face, crop_size)
                faces.append(torch.as_tensor(face))
        if len(faces) > 0:
            return torch.stack(faces), all_dets, images
        return None, None, images

    def detect_faces(self, img_paths: Union[str, list, np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray]:
        image = self.read_image(img_paths)
        pimage = self.preprocess(image)
        with torch.inference_mode():
            conf, loc, landms = self.model(pimage)
        dets = self.postprocess(conf, loc, landms, image.shape[1:3])
        return dets, image

    def warp_and_crop_face(self, image: np.ndarray, landmarks: np.ndarray, crop_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
        self.lmk_transform.estimate(landmarks.reshape(-1, 2), self.ref_pts)
        return cv2.warpAffine(image, self.lmk_transform.params[:2, :], crop_size)

    def visualize(self, images: np.ndarray, all_dets: List[np.ndarray]) -> np.ndarray:
        detections = []

        for image, dets in zip(images, all_dets):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            boxes, scores, lmks = dets[:, :4].astype(int), dets[:, 4], dets[:, 5:].astype(int)

            if boxes is None:
                detections.append(image)
                continue

            for box, score, lmk in zip(boxes, scores, lmks):
                draw_box_and_landmark(image, box, f"{int(score*100):3d}%", lmk)
            detections.append(image)
        return detections


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='assets/test.jpg')
    parser.add_argument('--model', type=str, default='RetinaFace')
    parser.add_argument('--checkpoint', type=str, default='/home/sithu/checkpoints/FR/retinaface/mobilenet0.25_Final.pth')
    parser.add_argument('--conf_threshold', type=float, default=0.6)
    parser.add_argument('--nms_threshold', type=float, default=0.4)
    args = vars(parser.parse_args())
    source = args.pop('source')
    file_path = Path(source)

    inference = FaceDetectAlign(**args)

    if file_path.is_dir():
        image_paths = list(file_path.glob('*'))
    elif file_path.is_file():
        image_paths = str(file_path)
    else:
        raise FileNotFoundError(f"{file_path} does not exist.")
    
    dets, images = inference.detect_faces(image_paths)
    det_images = inference.visualize(images, dets)
    for i, image in enumerate(det_images):
        image = Image.fromarray(image[:, :, ::-1]).convert('RGB')
        image.show()
        
