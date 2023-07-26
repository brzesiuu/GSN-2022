import functools

import numpy as np
from pytorch_lightning import Callback
import wandb

from transforms import DatasetTransform
from utils.enums import KeypointsMap
from utils.losses import PCKLoss
from utils.visualization import visualize_keypoints


class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, train_samples, num_samples=32, keypoints_map=KeypointsMap.CocoPose,
                 denorm=DatasetTransform.IMAGE_NET_INVERSE):
        super().__init__()
        self.num_samples = num_samples

        self.val_imgs = val_samples['image']
        self.val_predictions = val_samples['keypoints_2d']

        self.train_imgs = train_samples['image']
        self.train_predictions = train_samples['keypoints_2d']

        self.keypoints_map = keypoints_map
        self.denorm = denorm

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        train_imgs = self.train_imgs.to(device=pl_module.device)

        preds_val = pl_module(val_imgs)
        preds_train = pl_module(train_imgs)

        images_val = []
        for pred, image in zip(preds_val['keypoints_2d'], val_imgs):
            image_norm = self.denorm.value(image)
            image_norm = np.moveaxis(image_norm.cpu().numpy(), (0, 1, 2), (2, 0, 1))
            image_norm = (255 * image_norm).astype(np.uint8)
            image_norm = visualize_keypoints(image_norm, pred, keypoints_map=self.keypoints_map.value)
            images_val.append(wandb.Image(image_norm))

        images_train = []
        for pred, image in zip(preds_train['keypoints_2d'], train_imgs):
            image_norm = self.denorm.value(image)
            image_norm = np.moveaxis(image_norm.cpu().numpy(), (0, 1, 2), (2, 0, 1))
            image_norm = (255 * image_norm).astype(np.uint8)
            image_norm = visualize_keypoints(image_norm, pred, keypoints_map=self.keypoints_map.value)
            images_train.append(wandb.Image(image_norm))

        images_val_preds = []
        for pred, image in zip(self.val_predictions, val_imgs):
            image_norm = self.denorm.value(image)
            image_norm = np.moveaxis(image_norm.cpu().numpy(), (0, 1, 2), (2, 0, 1))
            image_norm = (255 * image_norm).astype(np.uint8)
            image_norm = visualize_keypoints(image_norm, pred, keypoints_map=self.keypoints_map.value)
            images_val_preds.append(wandb.Image(image_norm))

        images_train_preds = []
        for pred, image in zip(self.train_predictions, train_imgs):
            image_norm = self.denorm.value(image)
            image_norm = np.moveaxis(image_norm.cpu().numpy(), (0, 1, 2), (2, 0, 1))
            image_norm = (255 * image_norm).astype(np.uint8)
            image_norm = visualize_keypoints(image_norm, pred, keypoints_map=self.keypoints_map.value)
            images_train_preds.append(wandb.Image(image_norm))

        heatmaps = []
        for idx in range(len(preds_train['heatmaps'][0])):
            heatmap = preds_train['heatmaps'][0, idx].detach().cpu().numpy()
            heatmap = (255 * heatmap).astype(np.uint8)
            heatmaps.append(wandb.Image(heatmap))
        trainer.logger.experiment.log({
            "train_predictions": images_train,
            "val_predictions": images_val,
            "heatmaps": heatmaps,
            "train_gt": images_train_preds,
            "val_gt": images_val_preds
        })


class PCKCallback(Callback):
    def __init__(self, val_samples, distance_threshold=15):
        super().__init__()
        self.val_imgs = val_samples['image']
        self.val_predictions = val_samples['keypoints_2d']
        self.loss = functools.partial(PCKLoss, distance_threshold=distance_threshold)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        preds = pl_module(val_imgs)
        pck_loss = self.loss(preds['keypoints_2d'], self.val_predictions)

        pl_module.log('PCK', pck_loss, prog_bar=True)
