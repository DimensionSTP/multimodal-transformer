import os

os.environ["CURL_CA_BUNDLE"] = ""

import hydra

from src.pipelines.unimodal_pipeline import pipeline


@hydra.main(config_path="configs/", config_name="unimodal.yaml")
def main(config):
    return pipeline(config)


if __name__ == "__main__":
    main()
