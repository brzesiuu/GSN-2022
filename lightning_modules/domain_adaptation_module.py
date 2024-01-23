import pytorch_lightning as pl
import torch


class DomainAdaptationModule(pl.LightningModule):
    def __init__(self, model, partial_optimizer, heatmap_loss, discriminator_loss, input_key, heatmaps_key,
                 lr=None):
        super().__init__()

        self.save_hyperparameters()
        self.model = model
        self.partial_optimizer = partial_optimizer
        self.heatmap_loss = heatmap_loss
        self.discriminator_loss = discriminator_loss
        self.input_key = input_key
        self.heatmaps_key = heatmaps_key
        self.lr = 1e-1 if lr is None else lr

    def forward(self, x):
        return self.model(x)

    def compute_discriminator_loss(self, x, y):
        return self.compute_discriminator_loss(x, y)

    def compute_heatmap_loss(self, x, y):
        return self.heatmap_loss(x, y)

    def compute_loss(self, heatmap_loss, discriminator_loss):
        return heatmap_loss - discriminator_loss

    def common_step(self, batch, batch_idx):
        x, heatmaps_target, labels_target = batch[self.input_key], batch[self.heatmaps_key], batch[self.labels_key]
        outputs = self(x)

        heatmaps = outputs[self.heatmaps_key]
        heatmap_loss = self.compute_heatmap_loss(heatmaps, heatmaps_target)

        labels = self.forward_discriminator(x)
        discriminator_loss = self.compute_discriminator_loss(labels, labels_target)
        return heatmap_loss, discriminator_loss, outputs

    def common_test_valid_step(self, batch, batch_idx):
        heatmap_loss, discriminator_loss, outputs = self.common_step(batch, batch_idx)
        return heatmap_loss, discriminator_loss

    def training_step(self, batch, batch_idx):
        heatmap_loss, discriminator_loss = self.common_test_valid_step(batch, batch_idx)
        total_loss = self.compute_loss(heatmap_loss, discriminator_loss)
        self.log('train_heatmap_loss', heatmap_loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_discriminator_loss', discriminator_loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        heatmap_loss, discriminator_loss = self.common_test_valid_step(batch, batch_idx)
        total_loss = self.compute_loss(heatmap_loss, discriminator_loss)
        self.log('val_heatmap_loss', heatmap_loss, prog_bar=True)
        self.log('val_discriminator_loss', discriminator_loss, prog_bar=True)
        self.log('val_total_loss', total_loss, prog_bar=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        heatmap_loss, discriminator_loss = self.common_test_valid_step(batch, batch_idx)
        total_loss = self.compute_loss(heatmap_loss, discriminator_loss)
        self.log('test_heatmap_loss', heatmap_loss, prog_bar=True)
        self.log('test_discriminator_loss', discriminator_loss, prog_bar=True)
        self.log('test_total_loss', total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = self.partial_optimizer(params=self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, factor=0.5, patience=3, verbose=True, cooldown=2
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_total_loss'
        }
