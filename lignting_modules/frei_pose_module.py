import pytorch_lightning as pl
import torch


class FreiPoseModule(pl.LightningModule):
    def __init__(self, model, partial_optimizer, loss, input_key, output_key):
        super().__init__()

        self.save_hyperparameters()
        self.model = model
        self.partial_optimizer = partial_optimizer
        self.loss = loss
        self.input_key = input_key
        self.output_key = output_key

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, x, y):
        return self.loss(x, y)

    def common_step(self, batch, batch_idx):
        x, y = batch[self.input_key], batch[self.output_key]
        outputs = self(x)
        output = outputs[self.output_key]
        loss = self.compute_loss(output, y)
        return loss, outputs

    def common_test_valid_step(self, batch, batch_idx):
        loss, outputs = self.common_step(batch, batch_idx)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_test_valid_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_test_valid_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_test_valid_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.partial_optimizer(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, factor=0.5, patience=20, verbose=True, threshold=0.00001
        )
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
            'monitor': 'val_loss'
        }
