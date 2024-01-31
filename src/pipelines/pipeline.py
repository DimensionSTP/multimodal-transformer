from hydra.utils import instantiate
from omegaconf import DictConfig

from pytorch_lightning import Trainer, seed_everything
import wandb

from ..utils.setup import SetUp
from ..tuners.multimodal_tuner import MultiModalTuner


def train(config: DictConfig,) -> None:

    if "seed" in config:
        seed_everything(config.seed)

    setup = SetUp(config)

    train_loader = setup.get_train_loader()
    val_loader = setup.get_val_loader()
    architecture = setup.get_architecture()
    callbacks = setup.get_callbacks()
    logger = setup.get_wandb_logger()

    logged_hparams = {}
    for key, value in config.architecture.items():
        if key != "_target_" and key != "model":
            logged_hparams[key] = value
    if "model" in config.architecture:
        for key, value in config.architecture["model"].items():
            if key != "_target_":
                logged_hparams[key] = value
    logged_hparams["batch_size"] = config.batch_size
    logged_hparams["epoch"] = config.epoch
    logged_hparams["seed"] = config.seed
    logger.log_hyperparams(logged_hparams)

    trainer: Trainer = instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    try:
        trainer.fit(
            model=architecture,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        wandb.alert(
            title="Training Complete",
            text="Training process has successfully finished.",
            level=wandb.AlertLevel.INFO,
        )
    except Exception as e:
        wandb.alert(
            title="Training Error", 
            text="An error occurred during training", 
            level=wandb.AlertLevel.ERROR
        )
        raise e

def test(config: DictConfig,) -> None:

    if "seed" in config:
        seed_everything(config.seed)

    setup = SetUp(config)

    test_loader = setup.get_test_loader()
    architecture = setup.get_architecture()
    callbacks = setup.get_callbacks()
    logger = setup.get_wandb_logger()

    logged_hparams = {}
    for key, value in config.architecture.items():
        if key != "_target_" and key != "model":
            logged_hparams[key] = value
    if "model" in config.architecture:
        for key, value in config.architecture["model"].items():
            if key != "_target_":
                logged_hparams[key] = value
    logged_hparams["batch_size"] = config.batch_size
    logged_hparams["epoch"] = config.epoch
    logged_hparams["seed"] = config.seed
    logger.log_hyperparams(logged_hparams)

    trainer: Trainer = instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    try:
        trainer.test(
            model=architecture, dataloaders=test_loader, ckpt_path=config.ckpt_path
        )
        wandb.alert(
            title="Testing Complete",
            text="Testing process has successfully finished.",
            level=wandb.AlertLevel.INFO,
        )
    except Exception as e:
        wandb.alert(
            title="Testing Error", 
            text="An error occurred during testing", 
            level=wandb.AlertLevel.ERROR
        )
        raise e

def tune(config: DictConfig,) -> None:

    if "seed" in config:
        seed_everything(config.seed)

    setup = SetUp(config)

    train_loader = setup.get_train_loader()
    val_loader = setup.get_val_loader()

    tuner: MultiModalTuner = instantiate(
        config.tuner, train_loader=train_loader, val_loader=val_loader
    )
    tuner()