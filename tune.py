from omegaconf import DictConfig
import hydra

from src.pipeline.pipeline import tune


@hydra.main(config_path="configs/", config_name="kemdy19_multimodal_multimodal_tune.yaml")
def main(config: DictConfig,) -> None:
    return tune(config)


if __name__ == "__main__":
    main()