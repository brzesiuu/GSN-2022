from pathlib import Path

from utils import PosePath

def convert_frei_pose(input_folder: str|Path, output_folder: str|Path) -> None:
    training_dir = PosePath(input_folder).joinpath('training', 'rgb')
    eval_dir = PosePath(input_folder).joinpath('evaluation', 'rgb')

    training_dir.parents[0].joinpath('')




