from enum import Enum

import numpy as np
import cv2 as cv
import torch

from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.transforms import transforms as standard_transforms

from utils import PosePath, file_utils
from decorators.conversion_decorators import heatmaps, keypoints_2d
from utils.enums import KeypointsMap


class FreiPoseDataset(Dataset):
    """
    Dataset for FreiPose experiment.
    """

    def __init__(self, folder_path, set_type='training', image_extension='.jpg',
                 transform=transforms.Compose([transforms.ToTensor()]), resize=192, original_size=224,
                 denorm=None, heatmaps_scale=None) -> None:
        """
        Class constructor
        """
        self._path = folder_path
        self._set_type = set_type
        self._scale = resize / original_size
        self._resize = resize
        self._heatmaps_scale = heatmaps_scale

        self._image_paths = PosePath(self._path).joinpath('training', 'rgb').pose_glob('*' + image_extension,
                                                                                       natsort=True, to_list=True)
        self._camera_matrix_paths = PosePath(self._path).joinpath('training', 'data').pose_glob('*_K.json',
                                                                                                natsort=True,
                                                                                                to_list=True)
        self._xyz_paths = PosePath(self._path).joinpath('training', 'data').pose_glob('*_xyz.json',
                                                                                      natsort=True, to_list=True)

        self._transform = transform if not isinstance(transform, Enum) else transform.value

        self.denorm = denorm
        self.keypoints_map = KeypointsMap.FreiPose

    def __len__(self):
        return 5000

    @keypoints_2d
    @heatmaps(gaussian_kernel=7)
    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        coords = np.array(file_utils.load_config(self._xyz_paths[idx]), dtype=np.float32)
        camera_matrix = np.array(file_utils.load_config(self._camera_matrix_paths[idx]), dtype=np.float32)

        image = cv.imread(str(self._image_paths[idx]))
        image = standard_transforms.Compose(
            [standard_transforms.ToTensor(), standard_transforms.Resize((self._resize, self._resize))])(image)
        return {
            'image': self._transform(image),
            'keypoints_3d_local': coords,
            'camera_matrix': camera_matrix,
            'scale': self._scale,
            'heatmaps_scale': self._heatmaps_scale
        }
