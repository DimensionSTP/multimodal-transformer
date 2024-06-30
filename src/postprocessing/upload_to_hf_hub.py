import dotenv

dotenv.load_dotenv(
    override=True,
)

from huggingface_hub import HfApi, HfFolder

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="huggingface.yaml",
)
def upload_to_hf_hub(
    config: DictConfig,
) -> None:
    save_dir = f"{config.connected_dir}/prepare_upload/{config.pretrained_model_name}/epoch={config.epoch}"
    api = HfApi()
    token = HfFolder.get_token()
    api.upload_folder(
        folder_path=save_dir,
        repo_id=f"{config.user_name}/{config.model_type}-{config.upload_tag}",
        repo_type="model",
        token=token,
    )


if __name__ == "__main__":
    upload_to_hf_hub()
