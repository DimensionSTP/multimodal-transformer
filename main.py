import json

import hydra
from omegaconf import OmegaConf, DictConfig

from pytorch_lightning.utilities.distributed import rank_zero_info

from src.pipelines.pipeline import train, test, tune


@hydra.main(config_path="configs/", config_name="multimodal_main.yaml")
def main(config: DictConfig,) -> None:
    if config.is_tuned:
        params = json.load(open(config.tuned_hparams_path, "rt", encoding="UTF-8"))
        config = OmegaConf.merge(config, params)
    rank_zero_info(OmegaConf.to_yaml(config))

    if config.mode == "train":
        return train(config)
    elif config.mode == "test":
        return test(config)
    elif config.mode == "tune":
        return tune(config)
    else:
        raise ValueError(f"Invalid execution mode: {config.mode}")


if __name__ == "__main__":
    main()