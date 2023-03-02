from dataclasses import dataclass

import torchvision.transforms
import torchvision.transforms as transforms


@dataclass
class FreiPoseConfig:
    """
    Class for storing configuration data for FreiPose experiments.
    """
    folder_path: str
    set_type: str = 'training'
    image_extension: str = '.jpg'
    transform: transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)),
                                                transforms.Normalize([0.3950, 0.4323, 0.2954],
                                                                     [0.1966, 0.1734, 0.1836])])
