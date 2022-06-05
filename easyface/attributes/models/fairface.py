import torch
from torch import nn, Tensor
from torchvision.models import resnet34


class FairFace(nn.Module):
    def __init__(self, num_classes: int = 18) -> None:
        super().__init__()
        self.model = resnet34()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


if __name__ == '__main__':
    model = FairFace(18)
    model.load_state_dict(torch.load('/home/sithu/checkpoints/facialattributes/fairface/res34_fairface.pth', map_location='cpu'))
    x = torch.randn(2, 3, 112, 112)
    y = model(x)
    print(y.shape)
    # pre_dict = torch.load('/home/sithu/checkpoints/facialattributes/fairface/res34_fair_align_multi_7_20190809.pt', map_location='cpu')
    # new_dict = {}
    # for k, v in pre_dict.items():
    #     new_dict['model.' + k] = v
    # torch.save(new_dict, '/home/sithu/checkpoints/facialattributes/fairface/res34_fairface.pth')