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


Link repo 
public_image_set is 109GB
Upload preprocessed dataframes to personal drive

## TODO: 
1. Write extract token script to attain CLIP embeddings for image and text data and save. Refer to Food101 code for how to do this. 
2. Replace image and text backbone with just CLIP, refer to Food101 code for how to use CLIP.



