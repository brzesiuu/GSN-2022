import os
from pathlib import Path

from mmpose.apis import init_model
import torch.nn as nn

from decorators.conversion_decorators import keypoints_2d


class MMPoseModel(nn.Module):
    def __init__(self, config_path):
        super(MMPoseModel, self).__init__()
        try:
            tmp = init_model(config_path)
        except FileNotFoundError:
            path = str(Path(os.getcwd()).joinpath(config_path))
            tmp = init_model(path)
        self._backbone = tmp.backbone
        self._head = tmp.head
        self._neck = tmp.neck if hasattr(tmp, 'neck') else None
        self._heatmaps_scale = 0.25

    @keypoints_2d
    def forward(self, x):
        x = self._backbone(x)
        if self._neck is not None:
            x = self._neck(x)
        output = self._head(x)
        return {'heatmaps': output,
                'heatmaps_scale': self._heatmaps_scale}
