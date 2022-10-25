import torch
from torch import optim, nn
from torch.nn import functional as F
from torchmetrics import MetricCollection, F1Score, Accuracy

import pytorch_lightning as pl


class MultiModalPlModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        average: str,
        lr: float,
        t_max: int,
        eta_min: float,
        interval: str,
    ):
        super(MultiModalPlModule, self).__init__()
        self.model = model
        self.lr = lr
        self.t_max = t_max
        self.eta_min = eta_min
        self.interval = interval

        metrics = MetricCollection(
            [
                F1Score(num_classes=num_classes, average=average),
                Accuracy(num_classes=num_classes, average=average),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(
        self,
        audio: torch.Tensor,
        audio_mask: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
    ):
        output = self.model(
            audio=audio, audio_mask=audio_mask, text=text, text_mask=text_mask
        )
        return output

    def step(self, batch):
        audio, audio_mask, text, text_mask, label = batch
        output = self(
            audio=audio, audio_mask=audio_mask, text=text, text_mask=text_mask
        )
        loss = F.cross_entropy(output, label)
        pred = torch.argmax(output, dim=1)
        return loss, pred, label

    def configure_optimizers(self):
        adam_w_optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            adam_w_optimizer, T_max=self.t_max, eta_min=self.eta_min
        )
        return {
            "optimizer": adam_w_optimizer,
            "lr_scheduler": {"scheduler": cosine_scheduler, "interval": self.interval},
        }

    def training_step(self, batch, batch_idx):
        loss, pred, label = self.step(batch)
        return {"loss": loss, "pred": pred, "label": label}

    def validation_step(self, batch, batch_idx):
        loss, pred, label = self.step(batch)
        return {"loss": loss, "pred": pred, "label": label}

    def test_step(self, batch, batch_idx):
        loss, pred, label = self.step(batch)
        return {"loss": loss, "pred": pred, "label": label}

    def training_step_end(self, outputs):
        metrics = self.train_metrics(outputs["pred"], outputs["label"])
        self.log(
            "train_loss", outputs["loss"], on_step=False, on_epoch=True, prog_bar=False
        )
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return sum(outputs["loss"]) / len(outputs["loss"])

    def validation_step_end(self, outputs):
        metrics = self.val_metrics(outputs["pred"], outputs["label"])
        self.log(
            "val_loss", outputs["loss"], on_step=False, on_epoch=True, prog_bar=False
        )
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return sum(outputs["loss"]) / len(outputs["loss"])

    def test_step_end(self, outputs):
        metrics = self.test_metrics(outputs["pred"], outputs["label"])
        self.log(
            "test_loss", outputs["loss"], on_step=False, on_epoch=True, prog_bar=False
        )
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return sum(outputs["loss"]) / len(outputs["loss"])

    def on_epoch_end(self):
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.test_metrics.reset()
