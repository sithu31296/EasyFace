import torch
import math
from torch import nn, Tensor


def l2_norm(x, dim=1):
    norm = torch.norm(x, 2, dim, keepdim=True)
    output = torch.div(x, norm)
    return output


class AdaFace(nn.Module):
    def __init__(self, embedding_size=512, num_classes=70722, m=0.4, h=0.333, s=64, t_alpha=0.01) -> None:
        super().__init__()
        self.m = m
        self.h = h
        self.s = s
        self.t_alpha = t_alpha
        self.eps = 1e-3

        self.kernel = nn.Parameter(torch.Tensor(embedding_size, num_classes))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        self.register_buffer('batch_mean', torch.ones(1)*20)
        self.register_buffer('batch_std', torch.ones(1)*100)
        # self.norm_layer = nn.BatchNorm1d(1, eps=self.eps, momentum=self.t_alpha, affine=False)

    def forward(self, feats: Tensor, norms: Tensor, label: Tensor) -> Tensor:
        kernel_norm = l2_norm(self.kernel, dim=0)
        cosine = torch.mm(feats, kernel_norm).clamp(-1+self.eps, 1-self.eps)
        safe_norms = torch.clip(norms, min=0.001, max=100)

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std = std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)    # 66% between -1, 1
        # margin_scaler = self.norm_layer(safe_norms)
        margin_scaler = margin_scaler * self.h                                          # 68% between -0.333, 0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)

        # eg. m=0.5, h:0.333
        # range
        #   (66% range)
        #   -1  -0.333  0.333   1   (margin_scaler)
        # -0.5  -0.166  0.166 0.5   (m * margin_scaler)

        # g_angular
        m_arc = torch.zeros(label.shape[0], cosine.shape[1], device=cosine.device).scatter_(1, label.view(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, self.eps, math.pi - self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.shape[0], cosine.shape[1], device=cosine.device).scatter_(1, label.view(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m