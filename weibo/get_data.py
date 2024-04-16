import os
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, default_collate, WeightedRandomSampler

from PIL import Image
import argparse

from transformers import AutoProcessor
import torchvision.transforms as transforms

class WeiboDataset(Dataset): 
    def __init__(self, args, data_type):
        self.args = args
        self.data = pd.read_csv(os.path.join(args.data_path, "{}_data.csv".format(data_type)))
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333)),
            transforms.RandomGrayscale(p=0.1),
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_idx = row.img_idx
        text = row.text
        label = int(row.label)

        img_dir_path = "nonrumor" if label else "rumor"
        img_path = os.path.join(self.args.data_path, img_dir_path+"_images", "{}.jpg".format(img_idx))
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if 'qmf' in self.args.model_type:
            return text, img, label, idx
        return text, img, label
    
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
    label_counts = dataset.data.label.value_counts()
    weights = 1.0 / label_counts
    return WeightedRandomSampler(weights, len(dataset))
        
def get_data(args):
    train_set = WeiboDataset(args, "train")
    val_set = WeiboDataset(args, "test")
    test_set = WeiboDataset(args, "test")
    return train_set, val_set, test_set

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../data/weibo")
    args = parser.parse_args()
    setattr(args, "model_type", "qmf")
    dataset = WeiboDataset(args, "test")
    
    # make dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.custom_collate_fn)
    batch = next(iter(dataloader))
    print(batch[0].shape, batch[1].shape, batch[2].shape, batch[3].shape if len(batch) == 4 else None)
