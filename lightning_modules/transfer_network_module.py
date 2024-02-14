import random

import pytorch_lightning as pl
import torch


class TransferNetworkModule(pl.LightningModule):
    def __init__(self, transfer_network, partial_optimizer, lr=None, pretrain=None, content_weight=1.0,
                 style_weight=1.0):
        super().__init__()

        self.save_hyperparameters()
        self.transfer_network = transfer_network
        self.partial_optimizer = partial_optimizer

        self.lr = 1e-1 if lr is None else lr

        if pretrain:
            self._apply_pretrained_weights(pretrain)

        self.automatic_optimization = False
        self.content_weight = content_weight
        self.style_weight = style_weight

    def _apply_pretrained_weights(self, pretrain):
        weights = torch.load(pretrain)['state_dict']
        model_dict = self.transfer_network.state_dict()

        pretrained_dict = {k: v for k, v in weights.items() if k in model_dict}
        self.transfer_network.load_state_dict(pretrained_dict, strict=False)

    def forward(self, content, style):
        return self.transfer_network(content, style)

    def compute_loss(self, loss_style, loss_content):
        return self.style_weight * loss_style + self.content_weight * loss_content

    def common_step(self, batch, batch_idx):
        source_images = batch["train_batch"]
        target_images = batch["target_batch"]

        if random.random() > 0.5:
            content_images = source_images
            style_images = target_images
        else:
            content_images = target_images
            style_images = source_images

        loss_content, loss_style, g_t = self.transfer_network(content_images, style_images)
        return loss_content, loss_style, g_t

    def common_test_valid_step(self, batch, batch_idx):
        loss_content, loss_style, _ = self.common_step(batch, batch_idx)
        return loss_content, loss_style

    def manual_backward(self, loss, *args, **kwargs) -> None:
        loss.backward()

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()

        self.transfer_network.train()

        optimizer = optimizers.optimizer

        optimizer.zero_grad()

        loss_content, loss_style = self.common_test_valid_step(batch, batch_idx)
        total_loss = self.compute_loss(loss_content, loss_style)
        self.log('loss_content', loss_content, on_step=True, on_epoch=True, logger=True)
        self.log('loss_style', loss_style, on_step=True, on_epoch=True, logger=True)
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, logger=True)
        self.manual_backward(total_loss)

        optimizer.step()
        return total_loss

    def validation_step(self, batch, batch_idx):
        loss_content, loss_style = self.common_test_valid_step(batch, batch_idx)
        total_loss = self.compute_loss(loss_content, loss_style)
        self.log('val_content_loss', loss_content, prog_bar=True)
        self.log('val_style_loss', loss_style, prog_bar=True)
        self.log('val_total_loss', total_loss, prog_bar=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        loss_content, loss_style = self.common_test_valid_step(batch, batch_idx)
        total_loss = self.compute_loss(loss_content, loss_style)
        self.log('test_content_loss', loss_content, prog_bar=True)
        self.log('test_style_loss', loss_style, prog_bar=True)
        self.log('test_total_loss', total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = self.partial_optimizer(params=self.transfer_network.decoder.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, factor=0.5, patience=3, verbose=True, cooldown=2
        )
        return [{"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_total_loss"}]
