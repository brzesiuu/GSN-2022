import json
from enum import Enum

import numpy as np
import cv2 as cv
import torch

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from utils import PosePath, file_utils
from decorators.conversion_decorators import heatmaps, keypoints_2d


class CocoPoseDataset(Dataset):
    """
    Dataset for CocoPose experiment.
    """

    def __init__(self, json_path, image_path, set_type='training', image_extension='.jpg',
                 transform=transforms.Compose([transforms.ToTensor()]), scale=1) -> None:
        """
        Class constructor
        """
        self._json_paths = PosePath(json_path).pose_glob('*.json', natsort=True)

        self._set_type = set_type
        self._scale = scale
        self._transform = transform if not isinstance(transform, Enum) else transform.value
        self._base_shape = (192, 192)

    def __len__(self):
        return len(self._json_paths)

    @keypoints_2d
    @heatmaps
    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        data = file_utils.load_config(self._json_paths[idx])
        image = cv.imread(data["image_path"])
        keypoints = data["keypoints"]
        del keypoints[2::3]
        keypoints = np.array(keypoints).reshape((-1, 2)).astype(float)
        rows, cols = image.shape[:2]
        keypoints[:, 0] *= (self._base_shape[0] / cols)
        keypoints[:, 1] *= (self._base_shape[1] / rows)
        return {
            'image': self._transform(image),
            'keypoints_2d': keypoints,
            'scale': self._scale
        }
