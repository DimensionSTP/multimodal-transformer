import re

import pandas as pd

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer


class KEMDy19Dataset(Dataset):
    def __init__(self, data_path: str, pretrained_model: str, text_max_length: int):
        super(KEMDy19Dataset, self).__init__()
        self.data_path = data_path
        self.text_tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.text_max_length = text_max_length
        self.text, self.labels = self.load_data()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        normalized_text = self.normalize_string(self.text[idx])
        text_input = self.tokenize_text(normalized_text)
        text_data = {k: torch.tensor(v).squeeze() for k, v in text_input.items()}
        text_data["labels"] = torch.tensor(self.labels[idx])
        return text_data

    def tokenize_text(self, text):
        tokenized_text = self.text_tokenizer(
            text,
            max_length=self.text_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return tokenized_text

    @staticmethod
    def normalize_string(text):
        text = re.sub(r"[\s]", r" ", str(text))
        text = re.sub(r"[^a-zA-Z가-힣ㄱ-ㅎ0-9.!?]+", r" ", str(text))
        return text

    def load_data(self):
        data = pd.read_pickle(self.data_path)
        data = data.dropna()
        text = list(data["text"])
        labels = list(data["emotion"])
        return text, labels