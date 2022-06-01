import torch
from torch import nn


class FocalLoss(nn.Module):
    """Focal Loss

    """
    def __init__(self, gamma=0, eps=1e-7) -> None:
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = self.ce(pred, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()