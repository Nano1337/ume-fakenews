import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, default_collate
import pandas as pd
import numpy as np
import os
from PIL import Image

import argparse

from transformers import AutoProcessor

class FakedditDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.mode = mode
        df_path = os.path.join(args.data_path, f'{mode}__text_image__dataframe.pkl')
        df = pd.read_pickle(df_path)
        self.data_frame = df
        if args.num_classes == 2:
            self.label = "2_way_label"
        elif args.num_classes == 3:
            self.label = "3_way_label"
        elif args.num_classes == 6:
            self.label = "6_way_label"
    
    def __len__(self):
        return len(self.data_frame.index)

    def __getitem__(self, idx):

        # fetch modalities and label
        item_id = self.data_frame.loc[idx, 'id']
        label = torch.Tensor(
            [self.data_frame.loc[idx, self.label]]
        ).long().squeeze()
        text = self.data_frame.loc[idx, 'clean_title']
        image_path = os.path.join(self.args.image_dir_path, item_id + '.jpg')
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color = (0, 0, 0))

        # return contents depending on model type
        if 'qmf' in self.args.model_type: 
            return text, image, label, idx
        return text, image, label
    
    def custom_collate_fn(self, data):
        if 'qmf' in self.args.model_type:
            text, image, label, idx = zip(*data)
            label = default_collate(label)
            idx = default_collate(idx)
        else:
            text, image, label = zip(*data)
            label = default_collate(label)

        inputs = self.processor(text=text, images=image, padding="max_length", return_tensors="pt", truncation=True)
        text_tokens = inputs["input_ids"]
        img_tokens = inputs["pixel_values"]

        if 'qmf' in self.args.model_type:
            return text_tokens, img_tokens, label, idx
        else:   
            return text_tokens, img_tokens, label 
    
def get_sampler(dataset): 
    label_counts = dataset.data_frame[dataset.label].value_counts().to_dict()
    weights = [1.0 / label_counts[label] for label in dataset.data_frame[dataset.label]]
    return WeightedRandomSampler(weights, len(dataset))

def get_data(args):

    train_set = FakedditDataset(args, mode='train')
    val_set = FakedditDataset(args, mode='val')
    test_set = FakedditDataset(args, mode='test')

    return train_set, val_set, test_set

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    setattr(args, 'data_path', '../data/fakeddit/')
    setattr(args, 'image_dir_path', '/home/haoli/Documents/multimodal-clinical/data/fakenews/public_image_set')
    setattr(args, 'model_type', 'qmf')
    setattr(args, 'num_classes', 2)

    dataset = FakedditDataset(args, mode='train')
    
    loader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        persistent_workers=False,
        prefetch_factor=None, 
        collate_fn=dataset.collate_fn,
        sampler=get_sampler(dataset)
    )

    batch = next(iter(loader))



