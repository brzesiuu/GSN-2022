import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from datasets import FreiPoseDataset
from models import UNet


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    dataset = instantiate(cfg.dataset)
    model = instantiate(cfg.model)
    data = dataset[1]
    xd = model(data['image'].unsqueeze(0))
    print(5)


if __name__ == '__main__':
    main()
