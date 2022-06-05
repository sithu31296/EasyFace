import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.models import resnet18


class ConvBN(nn.Sequential):
    def __init__(self, c1, c2, k, s, p):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p),
            nn.BatchNorm2d(c2)
        )


class ChannelAttn(nn.Module):
    def __init__(self, c=512) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(c, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, c),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.gap(x)
        x = x.flatten(1)
        y = self.attention(x)
        return x * y


class SpatialAttn(nn.Module):
    def __init__(self, c=512) -> None:
        super().__init__()
        self.conv1x1 = ConvBN(c, 256, 1, 1, 0)
        self.conv_3x3 = ConvBN(256, c, 3, 1, 1)
        self.conv_1x3 = ConvBN(256, c, (1, 3), (1, 1), (0, 1))
        self.conv_3x1 = ConvBN(256, c, (3, 1), (1, 1), (1, 0))

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv1x1(x)
        y = F.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
        y = y.sum(dim=1, keepdim=True)
        return x * y


class CrossAttnHead(nn.Module):
    def __init__(self, c=512) -> None:
        super().__init__()
        self.sa = SpatialAttn(c)
        self.ca = ChannelAttn(c)
        self.apply(self._init_weights)

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self.ca(self.sa(x))



class DAN(nn.Module):
    def __init__(self, num_classes=7, pretrained: str = None) -> None:
        super().__init__()
        resnet = resnet18()
        if pretrained is not None:
            resnet.load_state_dict(torch.load(pretrained, map_location='cpu')['state_dict'])

        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.num_head = 4

        for i in range(self.num_head):
            setattr(self, f"cat_head{i}", CrossAttnHead())

        self.fc = nn.Linear(512, num_classes)
        self.bn = nn.BatchNorm1d(num_classes)

    def load_checkpoint(self, checkpoint: str = None) -> None:
        if checkpoint is not None:
            self.load_state_dict(torch.load(checkpoint, map_location='cpu')['model_state_dict'])

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self, f"cat_head{i}")(x))

        heads = torch.stack(heads).permute(1, 0, 2)
        heads = F.log_softmax(heads, dim=1)
        out = self.bn(self.fc(heads.sum(dim=1)))
        return out, x, heads


if __name__ == '__main__':
    model = DAN(7, '/home/sithu/checkpoints/face_emotion/resnet18_msceleb.pth')
    x = torch.randn(2, 3, 112, 112)
    out, x, heads = model(x)
    print(out.shape, x.shape, heads.shape)