import pytorch_lightning as pl


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
        loss = self.compute_loss(outputs, y)
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
        return self.partial_optimizer(self.model.parameters())
