# Unimodal Ensemble for Fake News Detection

Authors: Haoli, David, and Lincoln

This repository contains the code to reproduce all experiments for our paper in the AI4CPS class. A copy of the manuscript can be viewed ![here](Multimodal_Fake_News.pdf)

## Setup

This code was run in an Ubuntu 22.04 environment. Setup commands for other operating systems may differ.

1. Clone the repository with: 
```bash
git clone https://github.com/Nano1337/ume-fakenews.git
```

2. Create a virtual environment and activate it: 
```bash
python3 -m venv menv
source menv/bin/activate
```

3. version a: Accelerate setup by using the `uv` package: 
```bash
pip install uv
uv pip install -r requirements.txt
```

3. version b: Install the usual way required packages by running `pip install -r requirements.txt`

## Datasets

First, create a data folde  r from the root of the repository with `mkdir data`. 


### Fakeddit Dataset: 

1. Download the images and then decompress. Make sure you have gdown installed. NOTE: This dataset is 109GB.
```bash 
cd data/fakeddit
gdown 1cjY6HsHaSZuLVHywIxD5xQqng33J5S2b
tar -xjf public_images.tar.bz2
```

2. Download associated metadata from [here](https://drive.google.com/drive/folders/18WlBxUf_AHUlWGi4TYuVSdoQtdNht03_?usp=sharing). You can either manually download or use gdown again: 
```bash
cd data/fakeddit
gdown 1QMmD6Y7OpeGWNMiaqtwfEiXeWs3BB3iq
gdown 19mjxS1z6jZodhzkH1DyBO8gddqQXMHOh
gdown 1KGmyajyy054i4vhFpXqvE-ScphjiTeCn
```

3. Modify the config file `fakeddit/fakeddit.yaml` to point to the correct paths. Ensure the batch size fits within your GPU VRAM capacity. I'm running with a 24GB 4090 GPU, so I can use a batch size of 144 that uses about 22GB VRAM.

4. Start training the model with `python main.py --dir fakeddit`. 

### Weibo Dataset: 

1. Download the dataset using gdown, put it in the `data/weibo` folder, and decompress.
```bash
cd data/weibo
gdown 14VQ7EWPiFeGzxp3XC2DeEHi-BEisDINn
unzip weibo.zip
```

2. Download the processed annotation files from [here](https://drive.google.com/drive/folders/1QAD0BbqmHtElqt-pJWxsxdCUau1JW08R?usp=sharing) or use gdown: 
```bash
cd data/weibo
gdown 1oRosxoIjvYAlgWy4zIcUo8q4Ggx-lQCt
gdown 1qhrTwjFIJagC7mqqGp5VwsujLOkVJdou
```

3. Modify the config file `weibo/weibo.yaml` to point to the correct paths. Update the path and model type as needed.

4. Start training the model with `python main.py --dir weibo`.

## Reproduce Test Metrics

1. After you have trained the models, you can reproduce the test metrics by first finding the checkpoint file for the model you want to evaluate. This file should be in the `data/{dataset}/_ckpt` folder. 

2. Copy the absolute path of that checkpoint file and fill out the rest of the variables in the `generate_metrics.sh` file. You will also have to go into the `{dataset}.yaml` file and update the `model_type` variable to match the model you want to evaluate.

3. Run the script with `bash generate_metrics.sh`. This will print to the console the test metrics for the model you selected.

## Demo

To launch the demo, get the path of your trained checkpoint. Modify variables in `demo.sh` with the correct values. Then, run the following command: 
```bash
sh ./demo.sh
```
This will bring up a gradio demo where you can play with the model. 
