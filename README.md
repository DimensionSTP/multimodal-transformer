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

### Single Modality Training

* only audio
```shell
python unimodal_main.py mode=audio dataset=audio_kemdy19_dataset
```

* only text
```shell
python unimodal_main.py mode=text dataset=text_kemdy19_dataset
```

### Multi Modality Model Hyper-Parameters Tuning

* multimodal transformer(embedding vector deep fusion)
```shell
python main.py mode=tune is_tuned=untuned
```

### Multi Modality Training

* multimodal transformer(embedding vector deep fusion)
```shell
python main.py mode=train is_tuned={tuned or untuned}
```

### Multi Modality Testing

* multimodal transformer(embedding vector deep fusion)
```shell
python main.py mode=test is_tuned={tuned or untuned} epoch={ckpt epoch}
```


__If you want to change main config, use --config-name={config_name}.__

__Also, you can use --multirun option.__

__You can set additional arguments through the command line.__