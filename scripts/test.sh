#!/bin/bash

is_tuned="untuned"
strategy="ddp"
precision=32
batch_size=64
epochs="9 10"

for epoch in $epochs
do
    python main.py mode=test \
        is_tuned=$is_tuned \
        strategy=$strategy \
        precision=$precision \
        batch_size=$batch_size \
        epoch=$epoch
done
