import numpy as np
from pytorch_lightning import Callback
import wandb

from transforms import DatasetTransform
from utils.enums import KeypointsMap
from utils.visualization import visualize_keypoints


class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, num_samples=32, keypoints_map=KeypointsMap.FreiPose,
                 denorm=DatasetTransform.FREI_POSE_INVERSE):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs = val_samples['image']
        self.val_predictions = val_samples['keypoints_2d']
        self.keypoints_map = keypoints_map
        self.denorm = denorm

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)

        preds = pl_module(val_imgs)

        images = []
        for pred, image in zip(preds['keypoints_2d'], val_imgs):
            image_norm = self.denorm.value(image)
            image_norm = np.moveaxis(image_norm.cpu().numpy(), (0, 1, 2), (2, 1, 0))
            image_norm = (255 * image_norm).astype(np.uint8)
            image_norm = visualize_keypoints(image_norm, pred, keypoints_map=self.keypoints_map.value)
            images.append(wandb.Image(image_norm))
        trainer.logger.experiment.log({
            "examples": images
        })
