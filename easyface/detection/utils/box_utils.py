import torch
import math
from itertools import product


class PriorBox:
    def __init__(self, image_size=[112, 112]) -> None:
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.steps = [8, 16, 32]
        self.clip = False
        self.image_size = image_size
        self.feature_maps = [(math.ceil(self.image_size[0]/step), math.ceil(self.image_size[1]/step)) for step in self.steps]

    def __call__(self):
        anchors = []
        for min_sizes, step, feat in zip(self.min_sizes, self.steps, self.feature_maps):
            for i, j in product(range(feat[0]), range(feat[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * step / self.image_size[1] for x in [j+0.5]]
                    dense_cy = [y * step / self.image_size[0] for y in [i+0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        
        output = torch.Tensor(anchors).view(1, -1, 4)
        if self.clip:
            output.clamp_(0, 1)
        return output


def xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)
    """
    return torch.cat([boxes[:, :, :2] - boxes[:, :, 2:]/2, boxes[:, :, :2] + boxes[:, :, 2:]/2], dim=2)


def scale_coordinates(boxes: torch.Tensor, orig_size: list) -> torch.Tensor:
    boxes[:, :, ::2] *= orig_size[1]
    boxes[:, :, 1::2] *= orig_size[0]
    return boxes


def decode_boxes(boxes: torch.Tensor, priors: torch.Tensor, variances: list, orig_size: list) -> torch.Tensor:
    """Decode locations from predictions using priors
    to undo the encoding we did for offset regression at train time.

    boxes: lcoation predictions for loc layers, shape: [B, num_priors, 4]
    priors: prior boxes in center-offset form, shape: [1, num_priors, 4]
    variances: variances of prior boxes
    orig_size: original image shape [H, W]
    """
    boxes = torch.cat([
        priors[:, :, :2] + boxes[:, :, :2] * variances[0] * priors[:, :, 2:],
        priors[:, :, 2:] * torch.exp(boxes[:, :, 2:] * variances[1])
    ], dim=2)
    boxes = xywh2xyxy(boxes)
    boxes = scale_coordinates(boxes, orig_size)
    return boxes


def decode_landmarks(landmarks: torch.Tensor, priors: torch.Tensor, variances: list, orig_size: list) -> torch.Tensor:
    """Decode landmarks from predictions using priors
    to undo the encoding we did for offset regression at train time.

    landmakrs: landmarks predictions, shape: [B, num_priors, 10]
    priors: prior boxes in center-offset form, shape: [1, num_priors, 4]
    variances: variances of prior boxes
    orig_size: original image shape [H, W]
    """
    landmarks = torch.cat([
        priors[:, :, :2] + landmarks[:, :, :2] * variances[0] * priors[:, :, 2:],
        priors[:, :, :2] + landmarks[:, :, 2:4] * variances[0] * priors[:, :, 2:],
        priors[:, :, :2] + landmarks[:, :, 4:6] * variances[0] * priors[:, :, 2:],
        priors[:, :, :2] + landmarks[:, :, 6:8] * variances[0] * priors[:, :, 2:],
        priors[:, :, :2] + landmarks[:, :, 8:10] * variances[0] * priors[:, :, 2:]
    ], dim=2)
    landmarks = scale_coordinates(landmarks, orig_size)
    return landmarks