import hydra
import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

from lignting_modules.frei_pose_data_module import FreiPoseDataModule
from lignting_modules.frei_pose_module import FreiPoseModule


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    optimizer = instantiate(cfg.optimizer)
    dataset = instantiate(cfg.dataset)
    model = instantiate(cfg.model)

    train_module = FreiPoseModule(model, optimizer, torch.nn.MSELoss(), 'image',
                                  'heatmaps')
    data_module = FreiPoseDataModule(5, dataset, 0.9)

    trainer = pl.Trainer(max_epochs=50)

    trainer.fit(train_module, data_module)
    print(5)


if __name__ == '__main__':
    main()
