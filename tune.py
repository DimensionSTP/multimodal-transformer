from omegaconf import DictConfig
import hydra

from src.pipelines.pipeline import tune


@hydra.main(config_path="configs/", config_name="multimodal_tune.yaml")
def main(config: DictConfig,) -> None:
    return tune(config)


if __name__ == "__main__":
    main()