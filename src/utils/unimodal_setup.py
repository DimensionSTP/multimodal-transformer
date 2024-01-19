from typing import Union

from omegaconf import DictConfig
from hydra.utils import instantiate

from torch.utils.data import Dataset

from transformers import (
    Wav2Vec2FeatureExtractor,
    AutoTokenizer,
    HubertForSequenceClassification,
    AutoModelForSequenceClassification,
    TrainingArguments,
)
from datasets import load_metric
from datasets.metric import Metric


class SetUp:
    def __init__(self, config: DictConfig, modality: str,) -> None:
        self.config = config
        self.modality = modality

    def get_train_dataset(self) -> Dataset:
        train_dataset: Dataset = instantiate(
            self.config.dataset_module, data_path=self.config.data_path.train
        )
        return train_dataset

    def get_val_dataset(self) -> Dataset:
        val_dataset: Dataset = instantiate(
            self.config.dataset_module, data_path=self.config.data_path.val
        )
        return val_dataset

    def get_test_dataset(self) -> Dataset:
        test_dataset: Dataset = instantiate(
            self.config.dataset_module, data_path=self.config.data_path.test
        )
        return test_dataset

    def get_tokenizer(self) -> Union[AutoTokenizer, Wav2Vec2FeatureExtractor]:
        if self.modality == "audio":
            tokenizer = Wav2Vec2FeatureExtractor.from_pretrained(
                pretrained_model_name_or_path=self.config.pretrained_model_name,
            )
        elif self.modality == "text":
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.config.pretrained_model_name,
            )
        else:
            raise Exception("Only text or audio can be training modality")
        return tokenizer

    def get_model(
        self,
    ) -> Union[AutoModelForSequenceClassification, HubertForSequenceClassification]:
        if self.modality == "audio":
            model = HubertForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=self.config.pretrained_model_name,
                num_labels=self.config.num_labels,
                output_hidden_states=self.config.output_hidden_states,
            )
        elif self.modality == "text":
            model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=self.config.pretrained_model_name,
                num_labels=self.config.num_labels,
                output_hidden_states=self.config.output_hidden_states,
            )
        else:
            raise Exception("Only text or audio can be training modality")
        return model

    def get_metric(self) -> Metric:
        metric = load_metric(
            self.config.metric.first_metric, self.config.metric.second_metric
        )
        return metric

    def get_training_arguments(self) -> TrainingArguments:
        arguments: TrainingArguments = instantiate(self.config.training_arguments)
        return arguments
