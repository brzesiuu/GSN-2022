from torch import nn


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, output, target):
        losses = []
        for batch_pred, batch_target in zip(output, target):
            for heat_pred, heat_target in zip(batch_pred, batch_target):
                I = (heat_pred * heat_target).sum()
                U = (heat_pred * heat_pred).sum() + (heat_target * heat_target).sum() - (heat_pred * heat_target).sum()
                losses.append(1 - I / U)
        return sum(losses) / (len(output) * len(batch_pred))
