from transforms import HydraEnum
from torchvision.transforms import transforms as standard_transforms


class DatasetTransform(HydraEnum):
    FREI_POSE = standard_transforms.Compose([standard_transforms.Normalize([0.3950, 0.4323, 0.2954],
                                                                           [0.1966, 0.1734, 0.1836])])
    FREI_POSE_INVERSE = standard_transforms.Compose(
        [standard_transforms.Normalize([-0.3950 / 0.1966, -0.4323 / 0.1734, -0.2954 / 0.1836],
                                       [1 / 0.1966, 1 / 0.1734, 1 / 0.1836])])

    IMAGE_NET = standard_transforms.Compose([standard_transforms.Normalize([0.485, 0.456, 0.406],
                                                                           [0.229, 0.224, 0.225])])

    IMAGE_NET_INVERSE = standard_transforms.Compose(
        [standard_transforms.Normalize([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                       [1 / 0.229, 1 / 0.224, 1 / 0.225])])
