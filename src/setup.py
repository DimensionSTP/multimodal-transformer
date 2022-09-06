from typing import List

from omegaconf import DictConfig
from hydra.utils import instantiate

from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger


class SetUp:
    def __init__(self, config: DictConfig):
        self.config = config
        self.train_path = config.data_path.train
        self.val_path = config.data_path.val
        self.test_path = config.data_path.test

    def get_train_loader(self) -> DataLoader:
        train_dataset: Dataset = instantiate(
            self.config.dataset, data_path=self.train_path
        )
        return DataLoader(
            dataset=train_dataset, batch_size=self.config.batch_size, shuffle=True
        )

    def get_val_loader(self) -> DataLoader:
        val_dataset: Dataset = instantiate(self.config.dataset, data_path=self.val_path)
        return DataLoader(
            dataset=val_dataset, batch_size=self.config.batch_size, shuffle=False
        )

    def get_test_loader(self) -> DataLoader:
        test_dataset: Dataset = instantiate(
            self.config.dataset, data_path=self.test_path
        )
        return DataLoader(
            dataset=test_dataset, batch_size=self.config.batch_size, shuffle=False
        )

    def get_pl_module(self) -> pl.LightningModule:
        pl_module: pl.LightningModule = instantiate(self.config.pl_module)
        return pl_module

    def get_callbacks(self) -> List:
        model_checkpotint: ModelCheckpoint = instantiate(
            self.config.callbacks.model_checkpoint
        )
        early_stopping: EarlyStopping = instantiate(
            self.config.callbacks.early_stopping
        )
        return [model_checkpotint, early_stopping]

    def get_wandb_logger(self) -> WandbLogger:
        wandb_logger: WandbLogger = instantiate(self.config.logger.wandb)
        return wandb_logger
