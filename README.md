# Unimodal Ensemble for Fake News Detection

Authors: Haoli, David, and Lincoln

This repository contains the code to reproduce all experiments for our paper in the AI4CPS class. 

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

TODO: add testing script with precision, recall, and F1 score. Add documentation on how to test it
TODO: test late fusion, ogm-ge, qmf and run those checkpoints through the testing script too