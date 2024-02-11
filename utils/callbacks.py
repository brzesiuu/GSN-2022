import functools

import numpy as np
import torch
from pytorch_lightning import Callback
import wandb

from transforms import DatasetTransform
from utils.enums import KeypointsMap
from utils.losses import PCKLoss
from utils.visualization import visualize_keypoints


class ImagePredictionLoggerBase(Callback):
    def _get_prediction_visualizations(self, predictions, images, keypoints_map):
        output_images = []
        for pred, image in zip(predictions, images):
            image_norm = self.denorm.value(image)
            image_norm = np.moveaxis(image_norm.cpu().numpy(), (0, 1, 2), (2, 0, 1))
            image_norm = (255 * image_norm).astype(np.uint8)[:, :, ::-1]
            image_norm = visualize_keypoints(image_norm, pred, keypoints_map=keypoints_map.value)
            output_images.append(wandb.Image(image_norm))
        return output_images

    def _get_heatmap_visualizations(self, heatmaps):
        images = []
        for idx in range(len(heatmaps[0])):
            heatmap = heatmaps[0, idx].detach().cpu().numpy()
            heatmap = (255 * heatmap).astype(np.uint8)
            images.append(wandb.Image(heatmap))
        return images


class ImagePredictionLogger(ImagePredictionLoggerBase):
    def __init__(self, val_samples, train_samples, num_samples=32, keypoints_map=KeypointsMap.RenderedPose,
                 denorm=DatasetTransform.IMAGE_NET_INVERSE):
        super().__init__()
        self.num_samples = num_samples

        self.val_imgs = val_samples['image']
        self.val_predictions = val_samples['keypoints_2d']
        self.val_heatmaps = val_samples['heatmaps']

        self.train_imgs = train_samples['image']
        self.train_predictions = train_samples['keypoints_2d']
        self.train_heatmaps = train_samples['heatmaps']

        self.keypoints_map = keypoints_map
        self.denorm = denorm

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        train_imgs = self.train_imgs.to(device=pl_module.device)

        preds_val = pl_module(val_imgs)
        preds_train = pl_module(train_imgs)

        images_val = self._get_prediction_visualizations(preds_val['keypoints_2d'], val_imgs, self.keypoints_map)
        images_train = self._get_prediction_visualizations(preds_train['keypoints_2d'], train_imgs, self.keypoints_map)
        images_val_preds = self._get_prediction_visualizations(self.val_predictions, val_imgs, self.keypoints_map)
        images_train_preds = self._get_prediction_visualizations(self.train_predictions, train_imgs, self.keypoints_map)

        heatmaps = self._get_heatmap_visualizations(preds_train['heatmaps'])
        heatmaps_gt = self._get_heatmap_visualizations(self.train_heatmaps)

        trainer.logger.experiment.log({
            "train_predictions": images_train,
            "val_predictions": images_val,
            "heatmaps": heatmaps,
            "heatmaps_gt": heatmaps_gt,
            "train_gt": images_train_preds,
            "val_gt": images_val_preds
        })


class ImagePredictionLoggerDA(ImagePredictionLoggerBase):
    def __init__(self, val_samples, train_samples, num_samples=32, keypoints_map_source=KeypointsMap.RenderedPose,
                 keypoints_map_target=KeypointsMap.FreiPose, denorm=DatasetTransform.IMAGE_NET_INVERSE):
        super().__init__()
        self.num_samples = num_samples

        self.val_imgs = {'source': val_samples['train_batch']['image'],
                         'target': val_samples['target_batch']['image']}
        self.val_predictions = {'source': val_samples['train_batch']['keypoints_2d'],
                                'target': val_samples['target_batch']['keypoints_2d']}
        self.val_heatmaps = {'source': val_samples['train_batch']['heatmaps'],
                             'target': val_samples['target_batch']['heatmaps']}

        self.train_imgs = {'source': train_samples['train_batch']['image'],
                           'target': train_samples['target_batch']['image']}
        self.train_predictions = {'source': train_samples['train_batch']['keypoints_2d'],
                                  'target': train_samples['target_batch']['keypoints_2d']}
        self.train_heatmaps = {'source': train_samples['train_batch']['heatmaps'],
                               'target': train_samples['target_batch']['heatmaps']}

        self.keypoints_map_source = keypoints_map_source
        self.keypoints_map_target = keypoints_map_target
        self.denorm = denorm

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs_source = self.val_imgs['source'].to(device=pl_module.device)
        val_imgs_target = self.val_imgs['target'].to(device=pl_module.device)
        train_imgs_source = self.train_imgs['source'].to(device=pl_module.device)
        train_imgs_target = self.train_imgs['target'].to(device=pl_module.device)

        preds_val_source = pl_module(val_imgs_source)
        preds_val_target = pl_module(val_imgs_target)
        preds_train_source = pl_module(train_imgs_source)
        preds_train_target = pl_module(train_imgs_target)

        images_val_source = self._get_prediction_visualizations(preds_val_source['keypoints_2d'], val_imgs_source,
                                                                self.keypoints_map_source)
        images_train_source = self._get_prediction_visualizations(preds_train_source['keypoints_2d'], train_imgs_source,
                                                                  self.keypoints_map_source)
        images_val_source_gt = self._get_prediction_visualizations(self.val_predictions["source"],
                                                                   val_imgs_source, self.keypoints_map_source)
        images_train_source_gt = self._get_prediction_visualizations(self.train_predictions["source"],
                                                                     train_imgs_source, self.keypoints_map_source)

        images_val_target = self._get_prediction_visualizations(preds_val_target['keypoints_2d'], val_imgs_target,
                                                                self.keypoints_map_target)
        images_train_target = self._get_prediction_visualizations(preds_train_target['keypoints_2d'], train_imgs_target,
                                                                  self.keypoints_map_target)
        images_val_target_gt = self._get_prediction_visualizations(self.val_predictions["target"],
                                                                   val_imgs_target, self.keypoints_map_target)
        images_train_target_gt = self._get_prediction_visualizations(self.train_predictions["target"],
                                                                     train_imgs_target, self.keypoints_map_target)

        heatmaps_source = self._get_heatmap_visualizations(preds_train_source['heatmaps'])
        heatmaps_target = self._get_heatmap_visualizations(preds_train_target['heatmaps'])

        trainer.logger.experiment.log({
            "train_predictions_source": images_train_source,
            "val_predictions_source": images_val_source,
            "train_predictions_source_gt": images_train_source_gt,
            "val_predictions_source_gt": images_val_source_gt,
            "train_predictions_target": images_train_target,
            "val_predictions_target": images_val_target,
            "train_predictions_target_gt": images_train_target_gt,
            "val_predictions_target_gt": images_val_target_gt,
            "heatmaps_source": heatmaps_source,
            "heatmaps_target": heatmaps_target,
        })


class PCKCallback(Callback):
    def __init__(self, val_dataset, distance_threshold=15):
        super().__init__()
        self.val_dataset = val_dataset
        self.loss = functools.partial(PCKLoss, distance_threshold=distance_threshold)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        pck = []
        for batch in self.val_dataset:
            imgs = batch['image']
            gt = batch['keypoints_2d']
            val_imgs = imgs.to(device=pl_module.device)
            preds = pl_module(val_imgs)
            pck_loss = self.loss(preds['keypoints_2d'], gt)
            pck.append(pck_loss)
        pl_module.log('PCK', np.mean(pck), prog_bar=True)


class PCKCallbackDA(Callback):
    def __init__(self, val_dataset, distance_threshold=15):
        super().__init__()
        self.val_dataset = val_dataset
        self.loss = functools.partial(PCKLoss, distance_threshold=distance_threshold)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        pck = []
        for batch in self.val_dataset:
            imgs = batch['train_batch']['image']
            gt = batch['train_batch']['keypoints_2d']
            val_imgs = imgs.to(device=pl_module.device)
            preds = pl_module(val_imgs)
            pck_loss = self.loss(preds['keypoints_2d'], gt)
            pck.append(pck_loss)
        pl_module.log('PCK', np.mean(pck), prog_bar=True)


class StylePredictionLogger(Callback):
    def __init__(self, val_samples, train_samples, denorm=DatasetTransform.IMAGE_NET_INVERSE):
        super().__init__()

        self.val_imgs_source = val_samples['train_batch']
        self.val_imgs_target = val_samples['target_batch']

        self.train_imgs_source = train_samples['train_batch']
        self.train_imgs_target = train_samples['target_batch']

        self.denorm = denorm

    def _get_prediction_visualizations(self, images):
        output_images = []
        for image in images:
            image_norm = torch.clone(image)
            if self.denorm is not None:
                image_norm = self.denorm.value(image)
            image_norm = np.moveaxis(image_norm.cpu().numpy(), (0, 1, 2), (2, 0, 1))
            image_norm = (255 * image_norm).astype(np.uint8)[:, :, ::-1]
            output_images.append(wandb.Image(image_norm))
        return output_images

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs_source = self.val_imgs_source.to(device=pl_module.device)
        val_imgs_target = self.val_imgs_target.to(device=pl_module.device)
        train_imgs_source = self.train_imgs_source.to(device=pl_module.device)
        train_imgs_target = self.train_imgs_target.to(device=pl_module.device)

        _, _, preds_val_source = pl_module(val_imgs_source, val_imgs_target)
        _,_, preds_train_source = pl_module(train_imgs_source, train_imgs_target)

        images_val_target = self._get_prediction_visualizations(val_imgs_target)
        images_train_target = self._get_prediction_visualizations(train_imgs_target)

        images_val_source = self._get_prediction_visualizations(preds_val_source)
        images_train_source = self._get_prediction_visualizations(preds_train_source)

        trainer.logger.experiment.log({
            "train_predictions_source": images_train_source,
            "val_predictions_source": images_val_source,
            "train_target": images_train_target,
            "val_target": images_val_target,
        })
