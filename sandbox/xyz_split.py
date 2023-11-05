from utils import PosePath
import json

with open('C:\\datasets\\FreiPoseMini\\training_K.json', 'r') as f:
    K = json.load(f)
with open('C:\\datasets\\FreiPoseMini\\training_xyz.json', 'r') as f:
    xyz = json.load(f)

for idx, path in enumerate(
        PosePath('C:\\datasets\\FreiPoseMini').joinpath('training', 'rgb').pose_glob('*.jpg', natsort=True,
                                                                                     to_list=True)):
    K_name = path.name[:-4] + '_K.json'
    xyz_name = path.name[:-4] + '_xyz.json'

    if not path.parents[1].joinpath('data').exists():
        path.parents[1].joinpath('data').mkdir(parents=True, exist_ok=True)
    K_path = path.parents[1].joinpath('data', K_name)
    xyz_path = path.parents[1].joinpath('data', xyz_name)



    with open(K_path, 'w') as f:
        json.dump(K[idx % 32560], f)

    with open(xyz_path, 'w') as f:
        json.dump(xyz[idx % 32560], f)
