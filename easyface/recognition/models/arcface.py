import torch 
import torch.nn as nn
from torch import Tensor
from typing import Type, Optional, Union


class SEBlock(nn.Module):
    def __init__(self, ch: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch//reduction),
            nn.PReLU(),
            nn.Linear(ch//reduction, ch),
            nn.Sigmoid()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        B, C, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y


class IRBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_ch: int, out_ch: int, s: int = 1, downsample: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.bn0 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, s, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.prelu = nn.PReLU()

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample
        self.se = SEBlock(out_ch)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.prelu(self.bn1(self.conv1(self.bn0(x))))
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.prelu(out)

        return out


class ArcFace(nn.Module):
    def __init__(self, pretrained: str = None) -> None:
        super().__init__()
        self.inplanes = 64

        block = IRBlock
        layers = [2, 2, 2, 2]
        self.conv1 = nn.Conv2d(3, self.inplanes, 3, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

        self.layer1 = self._make_layer(block, 64, layers[0], s=1)
        self.layer2 = self._make_layer(block, 128, layers[1], s=2)
        self.layer3 = self._make_layer(block, 256, layers[2], s=2)
        self.layer4 = self._make_layer(block, 512, layers[3], s=2)

        self.bn4 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * 8 * 8 * block.expansion, 512)
        self.bn = nn.BatchNorm1d(512)

        self._init_weights(pretrained)


    def _init_weights(self, pretrained: str = None) -> None:
        if pretrained:
            try:
                self.load_state_dict(torch.load(pretrained, map_location='cpu'))
            except RuntimeError:
                pretrained_dict = torch.load(pretrained, map_location='cpu')
                pretrained_dict.popitem()   # remove bias
                pretrained_dict.popitem()   # remove weight
                self.load_state_dict(pretrained_dict, strict=False)
            finally:
                print(f"Loaded imagenet pretrained from {pretrained}")
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


    def _make_layer(self, block: IRBlock, planes: int, blocks: int, s: int = 1) -> nn.Sequential:
        downsample = None
        if s != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, s, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, s, downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.dropout(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = self.bn(x)
        return x



if __name__ == '__main__':
    model = ArcFace()
    x = torch.zeros(2, 3, 128, 128)
    y = model(x)
    print(y.shape)