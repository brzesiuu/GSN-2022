import torch
from torch import nn


class IoUEnergyLoss(nn.Module):
    def __init__(self, iou_weight=0.8):
        super(IoUEnergyLoss, self).__init__()
        self._eps = 1e-6

    def forward(self, output, target):
        iou_loss = IoUEnergyLoss()
        iou = iou_loss(output, target)

        output_energy = torch.sum(output, dim=0) / (output.shape[-1] * output.shape[-2])
        target_energy = torch.sum(target, dim=0) / (target.shape[-1] * target.shape[-2])

        return 1 - torch.mean(iou)
