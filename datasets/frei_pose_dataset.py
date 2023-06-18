from enum import Enum

import numpy as np
import cv2 as cv
import torch

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from decorators.debug_decorators import time_measure
from utils import PosePath, file_utils
from decorators.conversion_decorators import heatmaps, keypoints_2d


class FreiPoseDataset(Dataset):
    """
    Dataset for FreiPose experiment.
    """

    def __init__(self, folder_path, set_type='training', image_extension='.jpg',
                 transform=transforms.Compose([transforms.ToTensor()]), scale=[1, 1]) -> None:
        """
        Class constructor
        """
        self._path = folder_path
        self._set_type = set_type
        self._scale = scale

        self._image_paths = PosePath(self._path).joinpath('training', 'rgb').pose_glob('*' + image_extension,
                                                                                       natsort=True, to_list=True)
        self._camera_matrix_paths = PosePath(self._path).joinpath('training', 'data').pose_glob('*_K.json',
                                                                                                natsort=True,
                                                                                                to_list=True)
        self._xyz_paths = PosePath(self._path).joinpath('training', 'data').pose_glob('*_xyz.json',
                                                                                      natsort=True, to_list=True)

        self._transform = transform if not isinstance(transform, Enum) else transform.value

    def __len__(self):
        return len(self._image_paths)

    @keypoints_2d
    @heatmaps(gaussian_kernel=7)
    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        coords = np.array(file_utils.load_config(self._xyz_paths[idx]), dtype=np.float32)
        camera_matrix = np.array(file_utils.load_config(self._camera_matrix_paths[idx]), dtype=np.float32)

        image = cv.imread(str(self._image_paths[idx]))
        return {
            'image': self._transform(image),
            'keypoints_3d_local': coords,
            'camera_matrix': camera_matrix
        }
