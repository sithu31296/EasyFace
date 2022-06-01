import torch
from torch import nn, Tensor


class Flatten(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(1)


class BasicBlockIR(nn.Module):
    def __init__(self, c1, c2, s) -> None:
        super().__init__()
        if c1 == c2:
            self.shortcut_layer = nn.MaxPool2d(1, s)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, bias=False),
                nn.BatchNorm2d(c2)
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(c1),
            nn.Conv2d(c1, c2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.PReLU(c2),
            nn.Conv2d(c2, c2, 3, s, 1, bias=False),
            nn.BatchNorm2d(c2)
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class BottleneckIR(nn.Module):
    def __init__(self, c1, c2, s) -> None:
        super().__init__()
        ch = c2 // 4
        if c1 == c2:
            self.shortcut_layer = nn.MaxPool2d(1, s)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, bias=False),
                nn.BatchNorm2d(c2)
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(c1),
            nn.Conv2d(c1, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.PReLU(ch),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.PReLU(ch),
            nn.Conv2d(ch, c2, 1, s, 0, bias=False),
            nn.BatchNorm2d(c2)
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class AdaFace(nn.Module):
    def __init__(self, input_size: int = 112) -> None:
        super().__init__()
        assert input_size == 112
        channels = [64, 64, 128, 256, 512]

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        modules = []

        for i in range(4):
            modules.append(BasicBlockIR(channels[i], channels[i+1], 2))
            modules.append(BasicBlockIR(channels[i+1], channels[i+1], 1))

        self.body = nn.Sequential(*modules)

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512, affine=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        norm = torch.norm(x, 2, 1, True)
        out = torch.div(x, norm)
        return out


if __name__ == '__main__':
    model = AdaFace(112)
    model.load_state_dict(torch.load('/home/sithu/checkpoints/FR/adaface/adaface_ir18_webface4m.pth', map_location='cpu'), strict=False)
    x = torch.randn(2, 3, 112, 112)
    y = model(x)
    print(y.shape)
    
    # pre_dict = torch.load('/home/sithu/checkpoints/FR/adaface/adaface_ir18_webface4m.ckpt', map_location='cpu')['state_dict']
    # new_dict = {}

    # for k, v in pre_dict.items():
    #     if 'head' not in k:
    #         new_dict[k[6:]] = v
    #     else:
    #         new_dict[k] = v

    # torch.save(new_dict, '/home/sithu/checkpoints/FR/adaface/adaface_ir18_webface4m.pth')

    