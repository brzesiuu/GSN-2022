from abc import abstractmethod, ABC

from decorators.conversion_decorators import keypoints_2d


class BaseModel(ABC):
    def __init__(self):
        self._heatmaps_scale = None  # Needed for unification

    @abstractmethod
    def _forward_feature_extractor(self, x):
        pass

    @abstractmethod
    def _forward_heatmaps_extractor(self, x):
        pass

    def features(self, x):
        return self._forward_feature_extractor(x)

    @keypoints_2d
    def forward(self, x):
        x = self._forward_feature_extractor(x)
        output = self._forward_heatmaps_extractor(x)
        return {'heatmaps': output,
                'heatmaps_scale': self._heatmaps_scale}
