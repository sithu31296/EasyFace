import torch
import numpy as np
from torchvision.ops import nms


def torch_nms(boxes: torch.Tensor, scores: torch.Tensor, landmarks: torch.Tensor, conf_threshold, nms_threshold):
    # ignore low scores
    inds = scores >= conf_threshold
    scores = scores[inds]
    if scores.dim() == 0:   # no faces
        return None
    boxes = boxes[inds]
    landmarks = landmarks[inds]
    keep = nms(boxes, scores, nms_threshold)
    boxes = boxes[keep].view(-1, 4)
    scores = scores[keep].view(-1, 1)
    landmarks = landmarks[keep].view(-1, 10)
    return boxes, scores, landmarks


def py_cpu_nms(boxes, scores, landmarks, conf_threshold, nms_threshold):
    # ignore low scores
    inds = np.where(scores > conf_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]
    landmarks = landmarks[inds]

    # keep topk before NMS
    order = scores.argsort()[::-1][:5000]
    boxes = boxes[order]
    scores = scores[order]
    landmarks = landmarks[order]

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(overlap <= nms_threshold)[0]
        order = order[inds+1]

    boxes = boxes[keep, :]
    scores = scores[keep]
    landmarks = landmarks[keep, :]

    return boxes, scores, landmarks