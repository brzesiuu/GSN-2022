import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch

from lignting_modules.frei_pose_data_module import FreiPoseDataModule
from lignting_modules.frei_pose_module import FreiPoseModule
from utils.callbacks import ImagePredictionLogger, PCKCallback


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision('medium')
    config = instantiate(cfg)

    train_module = FreiPoseModule(config.model, config.optimizer, config.loss, config.input_key, config.output_key)
    data_module = FreiPoseDataModule(config.batch_size, config.dataset, config.train_ratio)

    val_samples = next(iter(data_module.val_dataloader()))

    config.trainer.callbacks.extend(
        [ImagePredictionLogger(val_samples), config.checkpoint_callback, PCKCallback(val_samples, 10)])
    config.trainer.logger = config.logger

    config.trainer.fit(train_module, data_module)

if __name__ == '__main__':
    main()
