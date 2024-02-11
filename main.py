import random

import hydra
import numpy as np
import omegaconf

import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
from pytorch_lightning.callbacks import LearningRateFinder, LearningRateMonitor

from lightning_modules.pose_data_module import PoseDataModule
from lightning_modules.pose_module import PoseModule
from utils.callbacks import ImagePredictionLogger, PCKCallback


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    model = hydra.core.hydra_config.HydraConfig.get().runtime.choices.model
    dataset = hydra.core.hydra_config.HydraConfig.get().runtime.choices.dataset
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    cfg["checkpoint_callback"][
        "dirpath"] = f"model/{dataset}_{model}"
    cfg["logger"]["name"] = "${now:%Y_%m_%d}_" + f"{model}_{len(dataset)}"

    config = instantiate(cfg)

    artifact = wandb.Artifact("hydra-configs", type="configs")
    artifact.add_dir(output_dir)
    wandb.log_artifact(artifact)

    train_module = PoseModule(config.model, config.optimizer, config.loss, config.input_key, config.output_key,
                              lr=config.lr)
    data_module = PoseDataModule(config.batch_size, config.dataset, config.train_ratio)

    val_samples = next(iter(data_module.val_dataloader()))
    train_samples = next(iter(data_module.train_dataloader()))
    lr_finder = LearningRateFinder(min_lr=1e-08, max_lr=1, num_training_steps=100, mode='exponential',
                                   early_stop_threshold=4.0, update_attr=True)
    pck_thresh = train_samples["image"].shape[2] * config.pck_ratio
    config.trainer.callbacks.extend(
        [lr_finder, ImagePredictionLogger(val_samples, train_samples, keypoints_map=config.dataset.keypoints_map,
                                          denorm=config.dataset.denorm), config.checkpoint_callback,
         PCKCallback(data_module.val_dataloader(), pck_thresh), LearningRateMonitor()])
    config.trainer.logger = config.logger

    config.trainer.fit(train_module, data_module)


if __name__ == '__main__':
    main()
