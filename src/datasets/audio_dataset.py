from typing import Dict, Any, List

import numpy as np
import pandas as pd
import librosa

import torch
from torch.utils.data import Dataset

from transformers import Wav2Vec2FeatureExtractor


class KEMDy19Dataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        target_column_name: str,
        pretrained_model: str,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.target_column_name = target_column_name
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            pretrained_model,
        )
        dataset = self.get_dataset()
        self.audio_paths = dataset["audio_paths"]
        self.labels = dataset["labels"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        audio = librosa.load(
            self.audio_paths[idx],
            sr=16000,
        )[0]
        audio_input = self.feature_extract_audio(audio)
        audio_data = {
            key: torch.tensor(val).squeeze() for key, val in audio_input.items()
        }
        audio_data["labels"] = torch.tensor(self.labels[idx])
        return audio_data

    def get_dataset(self) -> Dict[str, List[Any]]:
        data = pd.read_pickle(f"{self.data_path}/path_data/path_{self.split}.pkl")
        data = data.dropna()
        audio_paths = data["total_path"].values
        audio_paths = [
            f"{self.data_path}/{audio_path[2:]}" for audio_path in audio_paths
        ]
        labels = list(data[self.target_column_name])
        return {
            "audio_paths": audio_paths,
            "labels": labels,
        }

    def feature_extract_audio(
        self,
        audio: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        feature_extractor = self.feature_extractor
        input = feature_extractor(
            audio,
            sampling_rate=16000,
            max_length=80000,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return input
