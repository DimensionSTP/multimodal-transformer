import os

os.environ["CURL_CA_BUNDLE"] = ""

import hydra

from src.engine.only_text_engine import engine


@hydra.main(
    config_path="configs/", config_name="etri_basic_multimodal_only_text_main.yaml"
)
def main(config):
    return engine(config)


if __name__ == "__main__":
    main()
