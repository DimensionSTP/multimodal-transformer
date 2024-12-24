import dotenv

dotenv.load_dotenv(
    override=True,
)

import os
import warnings

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["HF_HOME"] = os.environ.get("HF_HOME")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import json

import torch

from tqdm import tqdm

import hydra
from omegaconf import OmegaConf, DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="multimodal.yaml",
)
def prepare_upload(
    config: DictConfig,
) -> None:
    if config.is_tuned == "tuned":
        params = json.load(
            open(
                config.tuned_hparams_path,
                "rt",
                encoding="UTF-8",
            )
        )
        config = OmegaConf.merge(
            config,
            params,
        )
    elif config.is_tuned == "untuned":
        pass
    else:
        raise ValueError(f"Invalid is_tuned argument: {config.is_tuned}")

    base_dir = f"{config.connected_dir}/prepare_upload/{config.model_detail}/{config.upload_tag}"
    checkpoints = [
        ckpt
        for ckpt in os.listdir(config.callbacks.model_checkpoint.dirpath)
        if ckpt.endswith(".ckpt")
    ]

    for ckpt in tqdm(checkpoints):
        save_dir = os.path.join(
            base_dir,
            ckpt.split(".")[0],
        )
        ckpt_path = os.path.join(
            config.callbacks.model_checkpoint.dirpath,
            ckpt,
        )

        os.makedirs(
            save_dir,
            exist_ok=True,
        )

        if config.strategy.startswith("deepspeed"):
            checkpoint = torch.load(f"{ckpt_path}/model.pt")
        else:
            checkpoint = torch.load(ckpt_path)

        model_save_path = os.path.join(
            save_dir,
            "model.ckpt",
        )
        torch.save(
            checkpoint,
            model_save_path,
        )


if __name__ == "__main__":
    prepare_upload()
