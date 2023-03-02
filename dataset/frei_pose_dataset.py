import numpy as np
import cv2 as cv
import torch

from torch.utils.data import Dataset

from . import FreiPoseConfig
from utils import PosePath, file_utils, conversion_utils


class FreiPoseDataset(Dataset):
    """
    Dataset for FreiPose experiment.
    """
    def __init__(self, config: FreiPoseConfig, device: str = 'cuda:0') -> None:
        """
        Class constructor
        :param config: Config with all necessary parameters for proper dataset creation
        :type config: FreiPoseConfig
        :param device: Device on which dataset will be placed during training
        :type device: str
        """
        self._device = torch.device(device)

        self._path = config.folder_path
        self._set_type = config.set_type

        self._image_paths = PosePath(self._path).joinpath('training', 'rgb').pose_glob('*' + config.image_extension,
                                                                                       natsort=True, to_list=True)
        self._camera_matrix_path = PosePath(self._path).joinpath(f'{self._set_type}_K.json')
        self._xyz_path = PosePath(self._path).joinpath(f'{self._set_type}_xyz.json')

        self._transform = config.transform

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        coords = np.array(file_utils.load_config(self._xyz_path)[idx % 32560], dtype=np.float32)
        camera_matrix = np.array(file_utils.load_config(self._camera_matrix_path)[idx % 32560], dtype=np.float32)

        image = cv.imread(str(self._image_paths[idx]))
        heatmaps = conversion_utils.get_heatmaps(coords, camera_matrix, image.shape)
        return self._transform(image), torch.Tensor(heatmaps)
