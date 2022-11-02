import os

os.environ["CURL_CA_BUNDLE"] = ""

import hydra

from src.engine.unimodal_engine import engine


@hydra.main(config_path="configs/", config_name="etri_unimodal_text_roberta_main.yaml")
def main(config):
    return engine(config, modality="text")


if __name__ == "__main__":
    main()
