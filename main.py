import random

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
from pytorch_lightning.callbacks import LearningRateFinder, LearningRateMonitor

from lignting_modules.frei_pose_data_module import FreiPoseDataModule
from lignting_modules.frei_pose_module import FreiPoseModule
from utils.callbacks import ImagePredictionLogger, PCKCallback


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    config = instantiate(cfg)
    train_module = FreiPoseModule(config.model, config.optimizer, config.loss, config.input_key, config.output_key,
                                  lr=config.lr)
    data_module = FreiPoseDataModule(config.batch_size, config.dataset, config.train_ratio)

    val_samples = next(iter(data_module.val_dataloader()))
    train_samples = next(iter(data_module.train_dataloader()))
    lr_finder = LearningRateFinder(min_lr=1e-08, max_lr=1, num_training_steps=100, mode='exponential',
                                   early_stop_threshold=4.0, update_attr=True)
    pck_thresh = train_samples["image"].shape[2] * config.pck_ratio
    config.trainer.callbacks.extend(
        [lr_finder, ImagePredictionLogger(val_samples, train_samples), config.checkpoint_callback,
         PCKCallback(data_module.val_dataloader(), pck_thresh), LearningRateMonitor()])
    config.trainer.logger = config.logger

    config.trainer.fit(train_module, data_module)

if __name__ == '__main__':
    main()
