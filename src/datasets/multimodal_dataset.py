from typing import Dict, Any, List
import re

import numpy as np
import pandas as pd

import librosa

import torch
from torch.utils.data import Dataset
from transformers import (
    Wav2Vec2FeatureExtractor,
    AutoTokenizer,
    HubertForSequenceClassification,
    AutoModelForSequenceClassification,
)


class KEMDy19Dataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        target_column_name: str,
        pretrained_hubert: str,
        pretrained_roberta: str,
        audio_max_length: int,
        text_max_length: int,
        num_labels: int,
        audio_conv_kernel: List[int],
        audio_conv_stride: List[int],
        device: str,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.target_column_name = target_column_name
        self.audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            pretrained_hubert,
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_roberta,
            use_fast=True,
        )
        self.audio_model = HubertForSequenceClassification.from_pretrained(
            pretrained_hubert,
            num_labels=num_labels,
            output_hidden_states=True,
        )
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_roberta,
            num_labels=num_labels,
            output_hidden_states=True,
        )
        self.audio_max_length = audio_max_length
        self.text_max_length = text_max_length
        self.audio_conv_kernel = audio_conv_kernel
        self.audio_conv_stride = audio_conv_stride
        self.device = device
        dataset = self.get_dataset()
        self.audio_paths = dataset["audio_paths"]
        self.texts = dataset["texts"]
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
        audio_data = {k: torch.tensor(v) for k, v in audio_input.items()}
        audio_data["labels"] = torch.tensor(self.labels[idx])
        audio_hidden = self.get_audio_hidden(audio_data)
        audio_mask = self.get_audio_padding_mask(audio_data["attention_mask"])

        normalized_text = self.normalize_string(self.texts[idx])
        text_input = self.tokenize_text(normalized_text)
        text_data = {k: torch.tensor(v) for k, v in text_input.items()}
        text_hidden = self.get_text_hidden(text_data)
        text_mask = self.get_text_padding_mask(text_data["attention_mask"])

        return {
            "audio_hidden": audio_hidden,
            "audio_mask": audio_mask,
            "text_hidden": text_hidden,
            "text_mask": text_mask,
            "label": self.labels[idx],
            "index": idx,
        }

    def get_dataset(self) -> Dict[str, List[Any]]:
        data = pd.read_pickle(f"{self.data_path}/path_data/path_{self.split}.pkl")
        data = data.dropna()
        audio_paths = data["total_path"].values
        audio_paths = [
            f"{self.data_path}/{audio_path[2:]}" for audio_path in audio_paths
        ]
        texts = list(data["text"])
        labels = list(data[self.target_column_name])
        return {
            "audio_paths": audio_paths,
            "texts": texts,
            "labels": labels,
        }

    def feature_extract_audio(
        self,
        audio: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        feature_extracted_audio = self.audio_feature_extractor(
            audio,
            sampling_rate=16000,
            max_length=self.audio_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return feature_extracted_audio

    def get_audio_hidden(
        self,
        audio_data: torch.Tensor,
    ) -> torch.Tensor:
        audio_model = self.audio_model.to(self.device)
        audio_model.eval()
        with torch.no_grad():
            input = {k: v.to(self.device) for k, v in audio_data.items()}
            output = audio_model(**input)
            hidden = output.hidden_states[0].to("cpu")
            torch.cuda.empty_cache()
        return hidden.squeeze()

    def get_after_conv_length(
        self,
        input_length: int,
    ) -> int:
        def conv_out_length(
            input_length: int,
            kernel_size: int,
            stride: int,
        ) -> int:
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(self.audio_conv_kernel, self.audio_conv_stride):
            input_length = conv_out_length(
                input_length,
                kernel_size,
                stride,
            )
        return input_length

    def get_audio_padding_mask(
        self,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        output_length = self.get_after_conv_length(
            attention_mask.sum(-1).to(
                torch.long,
            )
        )
        batch_size = attention_mask.shape[0]
        attention_mask = torch.zeros(
            (batch_size, self.text_max_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask[
            (
                torch.arange(
                    attention_mask.shape[0],
                    device=attention_mask.device,
                ),
                output_length - 1,
            )
        ] = 1
        attention_mask = attention_mask.cumsum(-1).bool()
        return attention_mask.squeeze()

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

    def get_text_hidden(
        self,
        text_data: torch.Tensor,
    ) -> torch.Tensor:
        text_model = self.text_model.to(self.device)
        text_model.eval()
        with torch.no_grad():
            input = {k: v.to(self.device) for k, v in text_data.items()}
            output = text_model(**input)
            hidden = output.hidden_states[0].to("cpu")
            torch.cuda.empty_cache()
        return hidden.squeeze()

    def get_text_padding_mask(
        self,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        output_length = attention_mask.sum(-1).to(
            torch.long,
        )
        batch_size = attention_mask.shape[0]
        attention_mask = torch.zeros(
            (
                batch_size,
                self.text_max_length,
            ),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask[
            (
                torch.arange(
                    attention_mask.shape[0],
                    device=attention_mask.device,
                ),
                output_length - 1,
            )
        ] = 1
        attention_mask = attention_mask.cumsum(-1).bool()
        return attention_mask.squeeze()
