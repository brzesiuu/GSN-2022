import random

import hydra
import numpy as np
import omegaconf

import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
from pytorch_lightning.callbacks import LearningRateMonitor

from datasets.domain_adaptation_dataset import SourceTargetDataset
from lightning_modules.pose_data_module import PoseDataModule
from lightning_modules.transfer_network_module import TransferNetworkModule
from utils.callbacks import StylePredictionLogger


@hydra.main(version_base=None, config_path="configs", config_name="config_style_transfer")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    model = hydra.core.hydra_config.HydraConfig.get().runtime.choices.model
    target_dataset = hydra.core.hydra_config.HydraConfig.get().runtime.choices['dataset@target_dataset']
    train_dataset = hydra.core.hydra_config.HydraConfig.get().runtime.choices['dataset@train_dataset']
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    cfg["checkpoint_callback"][
        "dirpath"] = f"model/{target_dataset}_{train_dataset}_{model}"
    cfg["logger"]["name"] = "${now:%Y_%m_%d}_"

    cfg["checkpoint_callback"][
        "dirpath"] = f"model/{train_dataset}_{target_dataset}_{model}"
    cfg["logger"]["name"] = "${now:%Y_%m_%d}_" + f"{model}_{len(train_dataset)}"

    config = instantiate(cfg)

    artifact = wandb.Artifact("hydra-configs", type="configs")
    artifact.add_dir(output_dir)
    wandb.log_artifact(artifact)

    train_module = TransferNetworkModule(config.model, config.optimizer, config.lr, pretrain=cfg.get('pretrain'),
                                         content_weight=config.content_weight,
                                         style_weight=config.style_weight)

    dataset = SourceTargetDataset(config.train_dataset, config.target_dataset)
    data_module = PoseDataModule(config.batch_size, dataset, config.train_ratio)

    val_samples = next(iter(data_module.val_dataloader()))
    train_samples = next(iter(data_module.train_dataloader()))

    config.trainer.callbacks.extend(
        [StylePredictionLogger(val_samples, train_samples, denorm=config.train_dataset.denorm),
         config.checkpoint_callback, LearningRateMonitor()])
    config.trainer.logger = config.logger

    config.trainer.fit(train_module, data_module)


if __name__ == '__main__':
    main()
