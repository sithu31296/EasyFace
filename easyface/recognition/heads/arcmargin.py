import torch
import math
from torch import nn
from torch.nn import functional as F



class ArcMarginProduct(nn.Module):
    """Large Margin Arc Distance Implementation

    """
    def __init__(self, in_features, out_features, s=30.0, m=0.5, easy_margin=False) -> None:
        super().__init__()
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(pred), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(1e-6, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # convert label to one-hot
        one_hot = torch.zeros(cosine.shape, device=pred.device).scatter_(1, target.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output



class AddMarginProduct(nn.Module):
    """Large Margin Cosine Distance Implementation

    """
    def __init__(self, in_features, out_features, s=30.0, m=0.5) -> None:
        super().__init__()
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(pred), F.normalize(self.weight))
        phi = cosine - self.m

        # convert to one-hot label
        one_hot = torch.zeros(cosine.shape, device=target.device).scatter_(1, target.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class SphereProduct(nn.Module):
    """Large Margin Cosine Distance Implementation

    """
    def __init__(self, in_features, out_features, m=4) -> None:
        super().__init__()
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.lambdamin = 5.0
        self.iter = 0

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.iter += 1
        self.lamb = max(self.lambdamin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        cos_theta = F.linear(F.normalize(pred), F.normalize(self.weight)).clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / math.pi).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        norm_of_feature = torch.norm(pred, 2, 1)

        # convert one-hot label
        one_hot = torch.zeros(cos_theta.shape, device=cos_theta.device).scatter_(1, target.view(-1, 1), 1)

        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= norm_of_feature.view(-1, 1)
        return output