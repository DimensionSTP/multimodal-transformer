# Multimodal Transformer

## 2022 ETRI multimodal emotion classification paper competition

### Dataset
ETRI KEMDy19

### Quick setup

```bash
# clone project
git clone https://github.com/DimensionSTP/multimodal-transformer.git
cd multimodal-transformer

# [OPTIONAL] create conda environment
conda create -n myenv python=3.7
conda activate myenv

# install requirements
pip install -r requirements.txt
```

### Single Modal Training

* only audio
```shell
python unimodal_audio_main.py
```

* only text
```shell
python unimodal_text_main.py
```

### Multi Modal Model Hyper-Parameters Tuning

* multimodal transformer(embedding vector deep fusion)
```shell
python tune.py
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