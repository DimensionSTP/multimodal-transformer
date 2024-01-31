from hydra.utils import instantiate
from omegaconf import DictConfig

from pytorch_lightning import Trainer, seed_everything

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

    logger_config = {}
    for key, value in config.architecture.items():
        if key != "_target_" and key != "model":
            logger_config[key] = value
    if "model" in config.architecture:
        for key, value in config.architecture["model"].items():
            if key != "_target_":
                logger_config[key] = value
    logger_config["batch_size"] = config.batch_size
    logger_config["epoch"] = config.epoch
    logger_config["seed"] = config.seed
    logger.experiment.config.update(logger_config)

    trainer: Trainer = instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    trainer.fit(
        model=architecture,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

def test(config: DictConfig,) -> None:

    if "seed" in config:
        seed_everything(config.seed)

    setup = SetUp(config)

    test_loader = setup.get_test_loader()
    architecture = setup.get_architecture()
    callbacks = setup.get_callbacks()
    logger = setup.get_wandb_logger()

    logger_config = {}
    for key, value in config.architecture.items():
        if key != "_target_" and key != "model":
            logger_config[key] = value
    if "model" in config.architecture:
        for key, value in config.architecture["model"].items():
            if key != "_target_":
                logger_config[key] = value
    logger_config["batch_size"] = config.batch_size
    logger_config["epoch"] = config.epoch
    logger_config["seed"] = config.seed
    logger.experiment.config.update(logger_config)

    trainer: Trainer = instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    trainer.test(
        model=architecture, dataloaders=test_loader, ckpt_path=config.ckpt_path
    )

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