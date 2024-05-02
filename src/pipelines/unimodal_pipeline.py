from typing import Tuple, Dict
import os

from omegaconf import DictConfig

import numpy as np
import pandas as pd

from transformers import Trainer

from ..utils.unimodal_setup import SetUp


def pipeline(
    config: DictConfig,
) -> None:
    setup = SetUp(config)

    train_dataset = setup.get_train_dataset()
    val_dataset = setup.get_val_dataset()
    test_dataset = setup.get_test_dataset()
    tokenizer = setup.get_tokenizer()
    model = setup.get_model()
    metric = setup.get_metric()
    training_arguments = setup.get_training_arguments()

    def compute_metrics(
        eval_pred: Tuple[np.ndarray, np.ndarray],
    ) -> Dict[str, float]:
        predictions, labels = eval_pred
        predictions = np.argmax(
            predictions,
            axis=1,
        )
        return metric.compute(
            predictions=predictions,
            references=labels,
        )

    trainer = Trainer(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        model=model,
        compute_metrics=compute_metrics,
        args=training_arguments,
    )

    trainer.train()
    trainer.evaluate()
    pred = trainer.predict(test_dataset)
    predictions = pred.predictions
    if not os.path.exists(config.save_predictions):
        os.makedirs(
            config.save_predictions,
            exist_ok=True,
        )
    np.save(
        f"{config.save_predictions}/{config.mode}.npy",
        predictions,
    )
    test = pd.read_pickle(config.data_path.test)
    test = test.dropna()
    y_pred = np.argmax(
        predictions,
        axis=1,
    )
    test["pred"] = y_pred
    labels = test.emotion.tolist()
    preds = test.pred.tolist()
    count = 0
    for i, j in zip(labels, preds):
        if i == j:
            count += 1
        else:
            pass
    acc = count / len(labels) * 100
    print(acc)
