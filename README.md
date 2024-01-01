# Multimodal Transformer

## 2022 ETRI multimodal emotion classification paper competition

### Dataset
ETRI KEMDy19

### Single Modal Training

* only audio
```shell
python unimodal_audio_main.py
```

* only text
```shell
python unimodal_text_main.py
```

### Multi Modal Training

* multimodal transformer(embedding vector deep fusion)
```shell
python train.py
```

### Multi Modal Testing

* multimodal transformer(embedding vector deep fusion)
```shell
python test.py
```