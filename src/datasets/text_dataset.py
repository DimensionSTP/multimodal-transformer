from typing import Dict, Any, List
import re

import pandas as pd

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer


class KEMDy19Dataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        pretrained_model: str,
        text_max_length: int,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model,
            use_fast=True,
        )
        self.text_max_length = text_max_length
        dataset = self.get_dataset()
        self.texts = dataset["texts"]
        self.labels = dataset["labels"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        normalized_text = self.normalize_string(self.texts[idx])
        text_input = self.tokenize_text(normalized_text)
        text_data = {k: torch.tensor(v).squeeze() for k, v in text_input.items()}
        text_data["labels"] = torch.tensor(self.labels[idx])
        return text_data

    def get_dataset(self) -> Dict[str, List[Any]]:
        data = pd.read_pickle(f"{self.data_path}/path_data/path_{self.split}.pkl")
        data = data.dropna()
        texts = list(data["text"])
        labels = list(data["emotion"])
        return {
            "texts": texts,
            "labels": labels,
        }

    @staticmethod
    def normalize_string(
        text: str,
    ) -> str:
        text = re.sub(
            r"[\s]",
            r" ",
            str(text),
        )
        text = re.sub(
            r"[^a-zA-Z가-힣ㄱ-ㅎ0-9.!?]+",
            r" ",
            str(text),
        )
        return text

    def tokenize_text(
        self,
        text: str,
    ) -> Dict[str, torch.Tensor]:
        tokenized_text = self.text_tokenizer(
            text,
            max_length=self.text_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return tokenized_text
