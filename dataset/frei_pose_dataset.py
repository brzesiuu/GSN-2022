from torch.utils.data import Dataset

from . import FreiPoseConfig
from utils import PosePath


class FreiPoseDataset(Dataset):
    def __init__(self, config: FreiPoseConfig):
        self._path = config.folder_path
        self._set_type = config.set_type

        self._image_paths = PosePath(self._path).joinpath('training', 'rgb').glob(config.image_extension, natsort=True,
                                                                                  to_list=True)
        self._camera_matrix_path = PosePath(self._path).joinpath(f'{self._set_type}_K.json')
        self._xyz_path = PosePath(self._path).joinpath(f'{self._set_type}_xyz.json')

        self._transform = config.transform

    def __getitem__(self, idx):
        pass


