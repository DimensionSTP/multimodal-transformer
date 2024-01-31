from typing import Tuple, Dict, Any

import torch
from torch import optim, nn
from torch.nn import functional as F
from torchmetrics import MetricCollection, F1Score, Accuracy

from pytorch_lightning import LightningModule


class MultiModalArchitecture(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        average: str,
        lr: float,
        t_max: int,
        eta_min: float,
        interval: str,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.t_max = t_max
        self.eta_min = eta_min
        self.interval = interval

        metrics = MetricCollection(
            [
                F1Score(task="multiclass", num_classes=num_classes, average=average),
                Accuracy(task="multiclass", num_classes=num_classes, average=average),
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
    ) -> torch.Tensor:
        output = self.model(
            audio=audio, audio_mask=audio_mask, text=text, text_mask=text_mask
        )
        return output

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio, audio_mask, text, text_mask, label = batch
        output = self(
            audio=audio, audio_mask=audio_mask, text=text, text_mask=text_mask
        )
        loss = F.cross_entropy(output, label)
        pred = torch.argmax(output, dim=1)
        return (loss, pred, label)

    def configure_optimizers(self) -> Dict[str, Any]:
        adam_w_optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            adam_w_optimizer, T_max=self.t_max, eta_min=self.eta_min
        )
        return {
            "optimizer": adam_w_optimizer,
            "lr_scheduler": {"scheduler": cosine_scheduler, "interval": self.interval},
        }

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int,) -> Dict[str, torch.Tensor]:
        loss, pred, label = self.step(batch)
        metrics = self.train_metrics(pred, label)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log_dict(
            metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return {"loss": loss, "pred": pred, "label": label}

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int,) -> Dict[str, torch.Tensor]:
        loss, pred, label = self.step(batch)
        metrics = self.val_metrics(pred, label)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log_dict(
            metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return {"loss": loss, "pred": pred, "label": label}

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int,) -> Dict[str, torch.Tensor]:
        loss, pred, label = self.step(batch)
        metrics = self.test_metrics(pred, label)
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log_dict(
            metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return {"loss": loss, "pred": pred, "label": label}

    def train_epoch_end(self, train_step_outputs: Dict[str, torch.Tensor],) -> None:
        self.train_metrics.reset()

    def validation_epoch_end(self, validation_step_outputs: Dict[str, torch.Tensor],) -> None:
        self.val_metrics.reset()

    def test_epoch_end(self, test_step_outputs: Dict[str, torch.Tensor],) -> None:
        self.test_metrics.reset()
