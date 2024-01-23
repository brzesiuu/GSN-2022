import os
from pathlib import Path

from mmpose.apis import init_model
import torch.nn as nn

from models.base_model import BaseModel


class MMPoseModel(nn.Module, BaseModel):
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

    def _forward_feature_extractor(self, x):
        output = self._backbone(x)
        if self._neck is not None:
            output = self._neck(output)
        return output

    def _forward_heatmaps_extractor(self, x):
        output = self._head(x)
        return output






