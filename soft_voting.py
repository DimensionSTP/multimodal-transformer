import os
import warnings

os.environ["HYDRA_FULL_ERROR"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="configs/",
    config_name="voting.yaml",
)
def softly_vote_logits(
    config: DictConfig,
) -> None:
    basic_path = config.basic_path
    submission_file = config.submission_file
    target_column_name = config.target_column_name
    voting_file = config.voting_file
    votings = config.votings

    weights = list(votings.values())
    if not np.isclose(sum(weights), 1):
        raise ValueError(f"summation of weights({sum(weights)}) is not equal to 1")

    weighted_logits = None
    for logit_file, weight in votings.items():
        try:
            logit = np.load(f"{basic_path}/logits/{logit_file}.npy")
        except:
            raise FileNotFoundError(f"logit file {logit_file} does not exist")
        if weighted_logits is None:
            weighted_logits = logit * weight
        else:
            weighted_logits += logit * weight

    ensemble_predictions = np.argmax(
        weighted_logits,
        axis=1,
    )
    submission_df = pd.read_csv(submission_file)
    submission_df[target_column_name] = ensemble_predictions
    submission_df.to_csv(
        voting_file,
        index=False,
    )


if __name__ == "__main__":
    softly_vote_logits()
