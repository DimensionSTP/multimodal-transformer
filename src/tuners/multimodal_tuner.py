from typing import Dict, Any
import os
import json

from torch.utils.data import DataLoader

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

from ..architectures.models.multimodal_transformer import MultiModalTransformer
from ..architectures.multimodal_architecture import MultiModalArchitecture


class MultiModalTuner:
    def __init__(
        self,
        hparams: Dict[str, Any],
        module_params: Dict[str, Any],
        direction: str,
        seed: int,
        num_trials: int,
        hparams_save_path: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: WandbLogger,
    ) -> None:
        self.hparams = hparams
        self.module_params = module_params
        self.direction = direction
        self.seed = seed
        self.num_trials = num_trials
        self.hparams_save_path = hparams_save_path
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger

    def __call__(self) -> None:
        study = optuna.create_study(
            direction=self.direction,
            sampler=TPESampler(seed=self.seed),
            pruner=HyperbandPruner(),
        )
        study.optimize(
            self.optuna_objective,
            n_trials=self.num_trials,
        )
        trial = study.best_trial
        best_score = trial.value
        best_params = trial.params
        print(f"Best score: {best_score}")
        print(f"Parameters: {best_params}")

        os.makedirs(
            self.hparams_save_path,
            exist_ok=True,
        )

        with open(f"{self.hparams_save_path}/best_params.json", "w") as json_file:
            json.dump(
                best_params,
                json_file,
            )

    def optuna_objective(
        self,
        trial: optuna.trial.Trial,
    ) -> float:
        seed_everything(self.seed)

        params = dict()
        params["seed"] = self.seed
        if self.hparams.num_heads:
            params["num_heads"] = trial.suggest_categorical(
                name="num_heads",
                choices=self.hparams.num_heads,
            )
        if self.hparams.num_layers:
            params["num_layers"] = trial.suggest_categorical(
                name="num_layers",
                choices=self.hparams.num_layers,
            )
        if self.hparams.attn_dropout:
            params["attn_dropout"] = trial.suggest_float(
                name="attn_dropout",
                low=self.hparams.attn_dropout.low,
                high=self.hparams.attn_dropout.high,
                log=self.hparams.attn_dropout.log,
            )
        if self.hparams.relu_dropout:
            params["relu_dropout"] = trial.suggest_float(
                name="relu_dropout",
                low=self.hparams.relu_dropout.low,
                high=self.hparams.relu_dropout.high,
                log=self.hparams.relu_dropout.log,
            )
        if self.hparams.res_dropout:
            params["res_dropout"] = trial.suggest_float(
                name="res_dropout",
                low=self.hparams.res_dropout.low,
                high=self.hparams.res_dropout.high,
                log=self.hparams.res_dropout.log,
            )
        if self.hparams.emb_dropout:
            params["emb_dropout"] = trial.suggest_float(
                name="emb_dropout",
                low=self.hparams.emb_dropout.low,
                high=self.hparams.emb_dropout.high,
                log=self.hparams.emb_dropout.log,
            )
        if self.hparams.out_dropout:
            params["out_dropout"] = trial.suggest_float(
                name="out_dropout",
                low=self.hparams.out_dropout.low,
                high=self.hparams.out_dropout.high,
                log=self.hparams.out_dropout.log,
            )
        if self.hparams.attn_mask:
            params["attn_mask"] = trial.suggest_categorical(
                name="attn_mask",
                choices=self.hparams.attn_mask,
            )
        if self.hparams.scale_embedding:
            params["scale_embedding"] = trial.suggest_categorical(
                name="scale_embedding",
                choices=self.hparams.scale_embedding,
            )
        if self.hparams.lr:
            params["lr"] = trial.suggest_float(
                name="lr",
                low=self.hparams.lr.low,
                high=self.hparams.lr.high,
                log=self.hparams.lr.log,
            )
        if self.hparams.weight_decay:
            params["weight_decay"] = trial.suggest_float(
                name="weight_decay",
                low=self.hparams.weight_decay.low,
                high=self.hparams.weight_decay.high,
                log=self.hparams.weight_decay.log,
            )
        if self.hparams.warmup_ratio:
            params["warmup_ratio"] = trial.suggest_float(
                name="warmup_ratio",
                low=self.hparams.warmup_ratio.low,
                high=self.hparams.warmup_ratio.high,
                log=self.hparams.warmup_ratio.log,
            )
        if self.hparams.eta_min_ratio:
            params["eta_min_ratio"] = trial.suggest_float(
                name="eta_min_ratio",
                low=self.hparams.eta_min_ratio.low,
                high=self.hparams.eta_min_ratio.high,
                log=self.hparams.eta_min_ratio.log,
            )

        model = MultiModalTransformer(
            model_dims=self.module_params.model_dims,
            num_heads=params["num_heads"],
            num_layers=params["num_layers"],
            audio_dims=self.module_params.model_dims,
            text_dims=self.module_params.model_dims,
            text_max_length=self.module_params.text_max_length,
            num_labels=self.module_params.num_labels,
            attn_dropout=params["attn_dropout"],
            relu_dropout=params["relu_dropout"],
            res_dropout=params["res_dropout"],
            emb_dropout=params["emb_dropout"],
            out_dropout=params["out_dropout"],
            attn_mask=params["attn_mask"],
            scale_embedding=params["scale_embedding"],
        )
        architecture = MultiModalArchitecture(
            model=model,
            num_labels=self.module_params.num_labels,
            average=self.module_params.average,
            strategy=self.module_params.strategy,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            warmup_ratio=params["warmup_ratio"],
            eta_min_ratio=params["eta_min_ratio"],
            interval=self.module_params.interval,
        )

        self.logger.log_hyperparams(params)
        callbacks = EarlyStopping(
            monitor=self.module_params.monitor,
            mode=self.module_params.mode,
            patience=self.module_params.patience,
            min_delta=self.module_params.min_delta,
        )

        trainer = Trainer(
            devices=self.module_params.devices,
            accelerator=self.module_params.accelerator,
            strategy=self.module_params.strategy,
            log_every_n_steps=self.module_params.log_every_n_steps,
            precision=self.module_params.precision,
            accumulate_grad_batches=self.module_params.accumulate_grad_batches,
            gradient_clip_val=self.module_params.gradient_clip_val,
            gradient_clip_algorithm=self.module_params.gradient_clip_algorithm,
            max_epochs=self.module_params.max_epochs,
            enable_checkpointing=False,
            callbacks=callbacks,
            logger=self.logger,
        )

        try:
            trainer.fit(
                model=architecture,
                train_dataloaders=self.train_loader,
                val_dataloaders=self.val_loader,
            )
            self.logger.experiment.alert(
                title="Tuning Complete",
                text="Tuning process has successfully finished.",
                level="INFO",
            )
        except Exception as e:
            self.logger.experiment.alert(
                title="Tuning Error",
                text="An error occurred during tuning",
                level="ERROR",
            )
            raise e

        return trainer.callback_metrics[self.module_params.monitor].item()
