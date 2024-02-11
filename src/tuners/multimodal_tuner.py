import os
from typing import Dict, Any, List
import json
import warnings

warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

from ..architectures.models.multimodal_transformer import MultiModalTransformer
from ..architectures.multimodal_architecture import MultiModalArchitecture


class MultiModalTuner:
    def __init__(
        self,
        hparams: Dict[str, Any],
        module_params: Dict[str, Any],
        num_trials: int,
        seed: int,
        hparams_save_path: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: WandbLogger,
    ) -> None:
        self.hparams = hparams
        self.module_params = module_params
        self.num_trials = num_trials
        self.hparams_save_path = hparams_save_path
        self.seed = seed
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger

    def __call__(self) -> None:
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.seed),
            pruner=HyperbandPruner(),
        )
        study.optimize(self.optuna_objective, n_trials=self.num_trials)
        trial = study.best_trial
        best_score = trial.value
        best_params = trial.params
        print(f"Best score : {best_score}")
        print(f"Parameters : {best_params}")

        if not os.path.exists(self.hparams_save_path):
            os.makedirs(self.hparams_save_path, exist_ok=True)

        with open(f"{self.hparams_save_path}/best_params.json", "w") as json_file:
            json.dump(best_params, json_file)

    def optuna_objective(
        self,
        trial: optuna.trial.Trial,
    ) -> float:
        seed_everything(self.seed)

        params = dict()
        params["seed"] = self.seed
        if self.hparams.n_heads:
            params["n_heads"] = trial.suggest_categorical(
                name="n_heads",
                choices=self.hparams.n_heads,
            )
        if self.hparams.n_layers:
            params["n_layers"] = trial.suggest_categorical(
                name="n_layers",
                choices=self.hparams.n_layers,
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
        if self.hparams.t_max:
            params["t_max"] = trial.suggest_int(
                name="t_max",
                low=self.hparams.t_max.low,
                high=self.hparams.t_max.high,
                log=self.hparams.t_max.log,
            )
        if self.hparams.eta_min:
            params["eta_min"] = trial.suggest_float(
                name="eta_min",
                low=self.hparams.eta_min.low,
                high=self.hparams.eta_min.high,
                log=self.hparams.eta_min.log,
            )

        model = MultiModalTransformer(
            d_model=self.module_params.d_model,
            n_heads=params["n_heads"],
            n_layers=params["n_layers"],
            d_audio=self.module_params.d_model,
            d_text=self.module_params.d_model,
            n_classes=self.module_params.num_classes,
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
            num_classes=self.module_params.num_classes,
            average=self.module_params.average,
            lr=params["lr"],
            t_max=params["t_max"],
            eta_min=params["eta_min"],
            interval=self.module_params.interval,
        )

        pruning_callback = PyTorchLightningPruningCallback(
            trial, monitor="val_MulticlassF1Score"
        )
        self.logger.log_hyperparams(params)

        trainer = Trainer(
            devices=1,
            accelerator=self.module_params.accelerator,
            log_every_n_steps=self.module_params.log_every_n_steps,
            precision=self.module_params.precision,
            max_epochs=self.module_params.max_epochs,
            enable_checkpointing=False,
            callbacks=pruning_callback,
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

        return trainer.callback_metrics["val_MulticlassF1Score"].item()
