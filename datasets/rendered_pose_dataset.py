from enum import Enum

import numpy as np
import cv2 as cv
import torch

from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.transforms import transforms as standard_transforms
import albumentations as A

from utils import PosePath, file_utils
from decorators.conversion_decorators import heatmaps, keypoints_2d
from utils.enums import KeypointsMap


class RenderedPoseDataset(Dataset):
    """
    Dataset for FreiPose experiment.
    """

    def __init__(self, folder_path, image_extension='.jpg',
                 transform=transforms.Compose([transforms.ToTensor()]), resize=192, original_size=320,
                 denorm=None, set_type='training',  heatmaps_scale=None) -> None:
        """
        Class constructor
        """
        self._path = folder_path
        self._scale = resize / original_size
        self._resize = resize
        self._heatmaps_scale = heatmaps_scale

        self._image_paths = PosePath(self._path).joinpath('images').pose_glob('*' + image_extension, natsort=True,
                                                                              to_list=True)

        self._annotation_paths = PosePath(self._path).joinpath('annotations').pose_glob('*.json',
                                                                                        natsort=True, to_list=True)

        self._transform = transform if not isinstance(transform, Enum) else transform.value
        self.denorm = denorm
        self.keypoints_map = KeypointsMap.RenderedPose
        self._set_type = set_type

    def __len__(self):
        return len(self._image_paths)

    @property
    def set_type(self):
        return self._set_type

    @keypoints_2d
    @heatmaps(gaussian_kernel=7)
    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        coords = np.array(file_utils.load_config(self._annotation_paths[idx]), dtype=np.float32)

        image = cv.imread(str(self._image_paths[idx]))
        if self.set_type == "training":
            image = standard_transforms.ToTensor()(A.Compose([A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
                               A.RandomBrightnessContrast(p=0.5),
                               A.Resize(self._resize, self._resize)])(image=image)["image"])
        else:
            image = standard_transforms.Compose(
                [standard_transforms.ToTensor(), standard_transforms.Resize((self._resize, self._resize))])(image)

        return {
            'image': self._transform(image),
            'keypoints_2d': self._scale * (coords),
            'scale': self._scale,
            'heatmaps_scale': self._heatmaps_scale
        }
