import torch
from torch import nn


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
        self._eps = 1e-6

    @staticmethod
    def _sum_dims(x):
        return x.sum(-1).sum(-1)

    def forward(self, output, target):
        intersection = self._sum_dims(output * target)
        union = self._sum_dims(output * output) + self._sum_dims(target * target) - self._sum_dims(output * target)
        iou = (intersection + self._eps) / (union + self._eps)
        return 1 - torch.mean(iou)
