from abc import abstractmethod

from torch import nn

from decorators.conversion_decorators import keypoints_2d


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def _forward(self, x):
        pass

    @keypoints_2d
    def forward(self, x):
        output = self._forward(x)
        return {'heatmaps': output,
                'heatmaps_scale': output[0, 0].shape[0] / x[0, 0].shape[0]}
