import torch
from torch import nn, Tensor


class ConvBNReLU(nn.Sequential):
    def __init__(self, c1, c2, k=3, s=1, p=1, leaky=0.):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(leaky, True)
        )


class DWConv(nn.Sequential):
    def __init__(self, c1, c2, s, leaky=0.1):
        super().__init__(
            nn.Conv2d(c1, c1, 3, s, 1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(leaky, True),
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(leaky, True)
        )


class MobileNetV1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBNReLU(3, 8, 3, 2, 1, 0.1),
            DWConv(8, 16, 1),
            DWConv(16, 32, 2),
            DWConv(32, 32, 1),
            DWConv(32, 64, 2),
            DWConv(64, 64, 1)
        )
        self.stage2 = nn.Sequential(
            DWConv(64, 128, 2),
            DWConv(128, 128, 1),
            DWConv(128, 128, 1),
            DWConv(128, 128, 1),
            DWConv(128, 128, 1),
            DWConv(128, 128, 1)
        )
        self.stage3 = nn.Sequential(
            DWConv(128, 256, 2),
            DWConv(256, 256, 1)
        )
        self.out_channels = [64, 128, 256]

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        return x1, x2, x3


if __name__ == '__main__':
    model = MobileNetV1()
    model.load_state_dict(torch.load('/home/sithu/checkpoints/FR/retinaface/mobilenetv1X0.25_imagenet.pth', map_location='cpu'), strict=False)
    x = torch.randn(2, 3, 112, 112)
    feats = model(x)
    for y in feats:
        print(y.shape)
    
    # pre_dict = torch.load('/home/sithu/checkpoints/FR/retinaface/mobilenetV1X0.25_pretrain.tar', map_location='cpu')['state_dict']
    # new_dict = {}
    # for k, v in pre_dict.items():
    #     new_dict[k[7:]] = v
    # torch.save(new_dict, '/home/sithu/checkpoints/FR/retinaface/mobilenetv1X0.25_imagenet.pth')