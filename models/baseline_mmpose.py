import os
from pathlib import Path

from mmpose.apis import init_model

from models.base_model import BaseModel


class MMPoseModel(BaseModel):
    def __init__(self, config_path):
        super(MMPoseModel, self).__init__()
        try:
            tmp = init_model(config_path)
        except FileNotFoundError:
            path = str(Path(os.getcwd()).joinpath(config_path))
            tmp = init_model(path)
        self.backbone = tmp.backbone
        self.keypoint_head = tmp.head
        self.neck = tmp.neck if hasattr(tmp, 'neck') else None

    def _forward_feature_extractor(self, x):
        output = self.backbone(x)
        if self.neck is not None:
            output = self.neck(output)
        return output

    def _forward_heatmaps_extractor(self, x):
        output = self.keypoint_head(x)
        return output

    def _forward(self, x):
        x = self._forward_feature_extractor(x)
        x = self._forward_heatmaps_extractor(x)
        return x
