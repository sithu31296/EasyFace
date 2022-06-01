import torch
from torch import nn, Tensor
from torch.nn import functional as F

from ..backbones import MobileNetV1


class ConvBNReLU(nn.Sequential):
    def __init__(self, c1, c2, k=3, s=1, p=1, leaky=0.):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(leaky, True)
        )


class ConvBN(nn.Sequential):
    def __init__(self, c1, c2, s):
        super().__init__(
            nn.Conv2d(c1, c2, 3, s, 1, bias=False),
            nn.BatchNorm2d(c2)
        )


class SSH(nn.Module):
    def __init__(self, c1, c2) -> None:
        super().__init__()
        leaky = 0.1 if c2 <= 64 else 0.

        self.conv3X3 = ConvBN(c1, c2//2, 1)
        self.conv5X5_1 = ConvBNReLU(c1, c2//4, 3, 1, 1, leaky)
        self.conv5X5_2 = ConvBN(c2//4, c2//4, 1)
        self.conv7X7_2 = ConvBNReLU(c2//4, c2//4, 3, 1, 1, leaky)
        self.conv7x7_3 = ConvBN(c2//4, c2//4, 1)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        conv3x3 = self.conv3X3(x)
        conv5x5_1 = self.conv5X5_1(x)
        conv5x5 = self.conv5X5_2(conv5x5_1)
        conv7x7_2 = self.conv7X7_2(conv5x5_1)
        conv7x7 = self.conv7x7_3(conv7x7_2)
        out = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)
        return self.relu(out)


class FPN(nn.Module):
    def __init__(self, in_channels, out_channel) -> None:
        super().__init__()
        leaky = 0.1 if out_channel <= 64 else 0.

        self.output1 = ConvBNReLU(in_channels[0], out_channel, 1, 1, 0, leaky)
        self.output2 = ConvBNReLU(in_channels[1], out_channel, 1, 1, 0, leaky)
        self.output3 = ConvBNReLU(in_channels[2], out_channel, 1, 1, 0, leaky)
        self.merge1 = ConvBNReLU(out_channel, out_channel, 3, 1, 1, leaky)
        self.merge2 = ConvBNReLU(out_channel, out_channel, 3, 1, 1, leaky)

    def forward(self, feats: Tensor) -> Tensor:
        output1 = self.output1(feats[0])
        output2 = self.output2(feats[1])
        output3 = self.output3(feats[2])

        up3 = F.interpolate(output3, size=output2.shape[-2:], mode='nearest')
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=output1.shape[-2:], mode='nearest')
        output1 = output1 + up2
        output1 = self.merge1(output1)
        return output1, output2, output3


class ClassHead(nn.Module):
    def __init__(self, ch=512, num_anchors=3) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(ch, num_anchors*2, 1)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.shape[0], -1, 2)
        return out


class BboxHead(nn.Module):
    def __init__(self, ch=512, num_anchors=3) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(ch, num_anchors*4, 1)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.shape[0], -1, 4)
        return out


class LandmarkHead(nn.Module):
    def __init__(self, ch=512, num_anchors=3) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(ch, num_anchors*10, 1)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.shape[0], -1, 10)
        return out


class RetinaFace(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channel = 64
        self.body = MobileNetV1()
        self.fpn = FPN(self.body.out_channels, channel)
        self.ssh1 = SSH(channel, channel)
        self.ssh2 = SSH(channel, channel)
        self.ssh3 = SSH(channel, channel)

        self.ClassHead = nn.ModuleList([
            ClassHead(channel, num_anchors=2)
        for _ in range(3)])

        self.BboxHead = nn.ModuleList([
            BboxHead(channel, num_anchors=2)
        for _ in range(3)])

        self.LandmarkHead = nn.ModuleList([
            LandmarkHead(channel, num_anchors=2)
        for _ in range(3)])

    def forward(self, x: Tensor) -> Tensor:
        feats = self.body(x)
        feats = self.fpn(feats)
        feat1 = self.ssh1(feats[0])
        feat2 = self.ssh2(feats[1])
        feat3 = self.ssh3(feats[2])
        features = [feat1, feat2, feat3]
        classifications, bbox_regress, landmark_regress = [], [], []

        for feat, ch, bh, lh in zip(features, self.ClassHead, self.BboxHead, self.LandmarkHead):
            classifications.append(ch(feat))
            bbox_regress.append(bh(feat))
            landmark_regress.append(lh(feat))

        classifications = torch.cat(classifications, dim=1)
        bbox_regress = torch.cat(bbox_regress, dim=1)
        landmark_regress = torch.cat(landmark_regress, dim=1)

        if self.training:
            return classifications, bbox_regress, landmark_regress
        return F.softmax(classifications, dim=-1), bbox_regress, landmark_regress


if __name__ == '__main__':
    model = RetinaFace()
    model.load_state_dict(torch.load('/home/sithu/checkpoints/FR/retinaface/mobilenet0.25_Final.pth', map_location='cpu'))
    x = torch.randn(2, 3, 112, 112)
    cls, bbox, lmk = model(x)
    print(cls.shape, bbox.shape, lmk.shape)