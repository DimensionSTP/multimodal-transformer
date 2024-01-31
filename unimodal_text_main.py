import os

os.environ["CURL_CA_BUNDLE"] = ""

import hydra

from src.pipelines.unimodal_pipeline import pipeline


@hydra.main(config_path="configs/", config_name="text_roberta_main.yaml")
def main(config):
    return pipeline(config, modality="text")


if __name__ == "__main__":
    main()
