import hydra
from omegaconf import OmegaConf, DictConfig

from pytorch_lightning.utilities.distributed import rank_zero_info

from src.pipeline.pipeline import test


@hydra.main(config_path="configs/", config_name="etri_basic_multimodal_test.yaml")
def main(config: DictConfig,) -> None:
    rank_zero_info(OmegaConf.to_yaml(config))
    return test(config)


if __name__ == "__main__":
    main()
