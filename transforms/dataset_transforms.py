from transforms import HydraEnum
from torchvision.transforms import transforms as standard_transforms


class DatasetTransform(HydraEnum):
    FREI_POSE = standard_transforms.Compose([standard_transforms.ToTensor(), standard_transforms.Resize((224, 224)),
                                             standard_transforms.Normalize([0.3950, 0.4323, 0.2954],
                                                                           [0.1966, 0.1734, 0.1836])])
    FREI_POSE_INVERSE = standard_transforms.Compose(
        [standard_transforms.Normalize([-0.3950 / 0.1966, -0.4323 / 0.1734, -0.2954 / 0.1836],
                                       [1 / 0.1966, 1 / 0.1734, 1 / 0.1836])])
