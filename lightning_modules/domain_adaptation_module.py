import pytorch_lightning as pl
import torch

from utils.optimizers import EMAOptimizer


class DomainAdaptationModule(pl.LightningModule):
    def __init__(self, teacher_model, student_model, partial_optimizer, heatmap_loss, input_key, heatmaps_key,
                 lr=None, style_net=None, pretrain=None):
        super().__init__()

        self.save_hyperparameters()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.style_net = style_net

        self.partial_optimizer = partial_optimizer
        self.heatmap_loss = heatmap_loss

        self.input_key = input_key
        self.heatmaps_key = heatmaps_key
        self.lr = 1e-1 if lr is None else lr

        self._teacher_optimizer_ema = EMAOptimizer(self.teacher_model, self.student_model)

        self._alpha = 1
        if pretrain:
            self._apply_pretrained_weights(pretrain)

        self.automatic_optimization = False

    def _apply_pretrained_weights(self, pretrain):
        weights = torch.load(pretrain)['state_dict']
        model_dict = self.teacher_model.state_dict()

        pretrained_dict = {k: v for k, v in weights.items() if k in model_dict}
        self.teacher_model.load_state_dict(pretrained_dict, strict=False)
        self.student_model.load_state_dict(pretrained_dict, strict=False)

    def forward(self, x):
        return self.student_model(x)

    def compute_discriminator_loss(self, x, y):
        return self.compute_discriminator_loss(x, y)

    def compute_heatmap_loss(self, x, y):
        return self.heatmap_loss(x, y)

    def compute_loss(self, heatmap_loss, teacher_loss):
        return heatmap_loss + self._alpha * teacher_loss

    def common_step(self, batch, batch_idx):
        train_batch = batch["train_batch"]
        target_batch = batch["target_batch"]
        x_source, heatmaps_source = train_batch[self.input_key], train_batch[self.heatmaps_key]
        x_target = target_batch[self.input_key]
        x_target_teacher = x_target.clone()
        if self.style_net is not None:
            x_source = self.style_net(x_source)
            x_target_teacher = self.style_net(x_target_teacher)
        outputs_source = self.student_model(x_source)
        outputs_target_student = self.student_model(x_target)
        outputs_target_teacher = self.teacher_model(x_target_teacher)

        heatmap_loss_source = self.compute_heatmap_loss(outputs_source[self.heatmaps_key], heatmaps_source)
        heatmap_loss_target = self.compute_heatmap_loss(outputs_target_student[self.heatmaps_key],
                                                        outputs_target_teacher[self.heatmaps_key])
        return heatmap_loss_source, heatmap_loss_target, outputs_source, outputs_target_student

    def common_test_valid_step(self, batch, batch_idx):
        heatmap_loss_source, heatmap_loss_target, _, _ = self.common_step(batch, batch_idx)
        return heatmap_loss_source, heatmap_loss_target

    def manual_backward(self, loss, *args, **kwargs) -> None:
        loss.backward()

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()

        self.teacher_model.train()
        self.student_model.train()

        student_opt = optimizers.optimizer

        student_opt.zero_grad()

        heatmap_loss_source, heatmap_loss_target = self.common_test_valid_step(batch, batch_idx)
        total_loss = self.compute_loss(heatmap_loss_source, heatmap_loss_target)
        self.log('train_heatmap_loss', heatmap_loss_source, on_step=True, on_epoch=True, logger=True)
        self.log('train_teacher_loss', heatmap_loss_target, on_step=True, on_epoch=True, logger=True)
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, logger=True)
        self.manual_backward(total_loss)

        student_opt.step()
        self._teacher_optimizer_ema.step()
        return total_loss

    def validation_step(self, batch, batch_idx):
        heatmap_loss_source, heatmap_loss_target = self.common_test_valid_step(batch, batch_idx)
        total_loss = self.compute_loss(heatmap_loss_source, heatmap_loss_target)
        self.log('val_heatmap_loss', heatmap_loss_source, prog_bar=True)
        self.log('val_teacher_loss', heatmap_loss_target, prog_bar=True)
        self.log('val_total_loss', total_loss, prog_bar=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        heatmap_loss_source, heatmap_loss_target = self.common_test_valid_step(batch, batch_idx)
        total_loss = self.compute_loss(heatmap_loss_source, heatmap_loss_target)
        self.log('test_heatmap_loss', heatmap_loss_source, prog_bar=True)
        self.log('test_teacher_loss', heatmap_loss_target, prog_bar=True)
        self.log('test_total_loss', total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        optimizer_student = self.partial_optimizer(params=self.student_model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer_student, factor=0.5, patience=3, verbose=True, cooldown=2
        )
        return [{"optimizer": optimizer_student, "lr_scheduler": scheduler, "monitor": "val_total_loss"}]
