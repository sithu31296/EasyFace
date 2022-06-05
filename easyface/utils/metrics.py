import torch



def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = torch.argmax(preds, dim=1)
    correct = torch.sum(preds == targets)
    return correct / len(targets)