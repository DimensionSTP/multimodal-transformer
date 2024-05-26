#!/bin/bash

is_tuned="untuned"
strategy="ddp"
precision=32
batch_size=64

python unimodal_main.py mode=audio \
    dataset=audio_kemdy19_dataset

python unimodal_main.py mode=text \
    dataset=text_kemdy19_dataset

python main.py mode=train \
    is_tuned=$is_tuned \
    strategy=$strategy \
    precision=$precision \
    batch_size=$batch_size
