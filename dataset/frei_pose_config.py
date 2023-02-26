from dataclasses import dataclass

import torchvision.transforms
import torchvision.transforms as transforms


@dataclass
class FreiPoseConfig:
    folder_path: str
    set_type: str = 'training'
    image_extension: str = '.jpg'
    transform: transforms = transforms.Compose([transforms.ToTensor])
