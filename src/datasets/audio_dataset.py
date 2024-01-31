from typing import Tuple, Dict, List, Any

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
        pretrained_model: str,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            pretrained_model
        )
        self.audio_path, self.labels = self.load_data()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int,) -> Dict[str, Any]:
        audio = librosa.load(self.audio_path[idx], sr=16000)[0]
        audio_input = self.feature_extract_audio(audio)
        item = {key: torch.tensor(val).squeeze() for key, val in audio_input.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def feature_extract_audio(self, audio: np.ndarray,) -> torch.Tensor:
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

    def load_data(self) -> Tuple[List[str], List[int]]:
        data = pd.read_pickle(self.data_path)
        data = data.dropna()
        split_current_path = self.data_path.split("/")[:3]
        current_path = (
            f"{split_current_path[0]}/{split_current_path[1]}/{split_current_path[2]}"
        )
        audio_path = data["total_path"].values
        audio_path = [f"{current_path}/{path[2:]}" for path in audio_path]
        labels = list(data["emotion"])
        return (audio_path, labels)
