import hydra
from omegaconf import OmegaConf

from pytorch_lightning.utilities.distributed import rank_zero_info

from src.pipeline.pipeline import train


@hydra.main(config_path="configs/", config_name="etri_basic_multimodal_train.yaml")
def main(config):
    rank_zero_info(OmegaConf.to_yaml(config))
    return train(config)


if __name__ == "__main__":
    main()
