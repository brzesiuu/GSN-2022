import random

import hydra
import numpy as np

import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
from pytorch_lightning.callbacks import LearningRateFinder, LearningRateMonitor

from datasets.domain_adaptation_dataset import DomainAdaptationDataset
from lightning_modules.domain_adaptation_module import DomainAdaptationModule
from lightning_modules.frei_pose_data_module import FreiPoseDataModule
from models.pose_da import MMPoseDAGAN
from utils.callbacks import ImagePredictionLogger, PCKCallback


@hydra.main(version_base=None, config_path="configs", config_name="config_da")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    model = hydra.core.hydra_config.HydraConfig.get().runtime.choices.model
    target_dataset = hydra.core.hydra_config.HydraConfig.get().runtime.choices['dataset@target']
    train_dataset = hydra.core.hydra_config.HydraConfig.get().runtime.choices['dataset@train']
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    cfg["checkpoint_callback"][
        "dirpath"] = f"model/{target_dataset}_{train_dataset}_{model}"
    cfg["logger"]["name"] = "${now:%Y_%m_%d}_"

    config = instantiate(cfg)

    artifact = wandb.Artifact("hydra-configs", type="configs")
    artifact.add_dir(output_dir)
    wandb.log_artifact(artifact)

    train_module = DomainAdaptationModule(MMPoseDAGAN(config.model), config.optimizer, config.heatmap_loss,
                                          config.discriminator_loss, config.input_key, config.output_key,
                                          lr=config.lr)

    dataset = DomainAdaptationDataset(config.train_dataset, config.target_dataset)
    data_module = FreiPoseDataModule(config.batch_size, dataset, config.train_ratio)

    val_samples = next(iter(data_module.val_dataloader()))
    train_samples = next(iter(data_module.train_dataloader()))
    lr_finder = LearningRateFinder(min_lr=1e-08, max_lr=1, num_training_steps=100, mode='exponential',
                                   early_stop_threshold=4.0, update_attr=True)
    pck_thresh = train_samples["image"].shape[2] * config.pck_ratio
    config.trainer.callbacks.extend(
        [lr_finder, ImagePredictionLogger(val_samples, train_samples, keypoints_map=config.train_dataset.keypoints_map,
                                          denorm=config.train_dataset.denorm), config.checkpoint_callback,
         PCKCallback(data_module.val_dataloader(), pck_thresh), LearningRateMonitor()])
    config.trainer.logger = config.logger

    config.trainer.fit(train_module, data_module)


if __name__ == '__main__':
    main()
