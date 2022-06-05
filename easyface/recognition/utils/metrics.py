import torch
import itertools
import numpy as np
from sklearn.metrics import roc_curve


def l2_sim(x, y):
    return 1. / (1 + np.linalg.norm(y - x))


def cos_sim(x, y):
    x, y = x.reshape(-1), y.reshape(-1)
    return np.dot(x, y) / (1e-10 + np.linalg.norm(x) * np.linalg.norm(y))


class Evaluator:
    def __init__(self) -> None:
        self.similarity = 'cos'
        self.limit = 50
        self.batch_size = 8

    def evaluate(self, embeddings: np.ndarray, labels: np.ndarray):
        unique_labels = np.unique(labels).astype(int)

        real_idxs = {label: np.random.permutation(np.argwhere(labels == label))[:self.limit] for label in unique_labels}
        fake_idxs = {label: np.random.permutation(np.argwhere(labels != label))[:self.limit] for label in unique_labels}

        real_scores, fake_scores = [], []

        for label in unique_labels:
            for i, j in itertools.combinations(real_idxs[label], r=2):
                score = cos_sim(embeddings[i], embeddings[j])
                real_scores.append(score)

            for i, j in itertools.product(real_idxs[label], fake_idxs[label]):
                score = cos_sim(embeddings[i], embeddings[j])
                fake_scores.append(score)

        real_scores = np.nan_to_num(np.array(real_scores))
        fake_scores = np.nan_to_num(np.array(fake_scores))

        fpr, tpr, threshold = roc_curve(np.hstack([np.ones(len(real_scores)), np.zeros(len(fake_scores))]), np.hstack([real_scores, fake_scores]))
        fnr = 1 - tpr

        eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

        return eer, eer_threshold