
import argparse
import os

from pytorch_lightning import seed_everything
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from utils.setup_configs import setup_configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="which directory to run")
    parser.add_argument("--dir", type=str, default=None, help="directory to run")
    parser.add_argument("--model_type", type=str, default=None, help="model type to run")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint to load")
    args = parser.parse_args()
    model_type = args.model_type
    ckpt = args.ckpt
    args = setup_configs(parser=parser)

    setattr(args, "model_type", str(args.model_type))
    setattr(args, "ckpt", str(args.ckpt))
    
    test_loader = None
    model = None

    if args.dir == "fakeddit":
        from fakeddit.get_data import FakedditDataset
        from fakeddit import get_model
        test_set = FakedditDataset(args, "test")
        setattr(args, "num_samples", len(test_set))
        test_loader = DataLoader(
            test_set, 
            batch_size=args.batch_size, 
            num_workers=args.num_cpus, 
            persistent_workers=True, 
            prefetch_factor=4,
            shuffle=False,
            collate_fn=test_set.custom_collate_fn, 
            pin_memory=True,
        )
        model = get_model(args)
    elif args.dir == "weibo":
        from weibo.get_data import WeiboDataset
        from weibo import get_model
        test_set = WeiboDataset(args, "test")
        setattr(args, "num_samples", len(test_set))
        test_loader = DataLoader(
            test_set, 
            batch_size=args.batch_size, 
            num_workers=args.num_cpus, 
            persistent_workers=True, 
            prefetch_factor=4,
            shuffle=False,
            collate_fn=test_set.custom_collate_fn, 
            pin_memory=True,
        )
        model = get_model(args)
    else: 
        raise NotImplementedError("No directory provided, please specify flag --dir")
    
    # load model ckpt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.ckpt:
        # load 
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        model.to(device)
    else: 
        raise NotImplementedError("No checkpoint provided, please specify flag --ckpt")

    # Initialize lists to store true labels and predictions
    true_labels = []
    predictions = []

    # run inference
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Running Inference"):
            if model_type == "qmf":
                text, image, label, idx = batch
                text, image, label = text.to(device), image.to(device), label.to(device)
            else: 
                text, image, label = batch 
                text, image, label = text.to(device), image.to(device), label.to(device)

            if model_type == "jlogits":
                _, _, logits, _ = model(text, image, label) 
            elif model_type == "ensemble":
                x1_logits, x2_logits, _, _ = model(text, image, label)
                logits = (x1_logits + x2_logits)/2
            elif model_type == "ogm_ge": 
                _, _, logits, _ = model(text, image, label)
            elif model_type == "qmf":
                _, _, _, _, logits = model(text, image, label, idx)
            else: 
                raise NotImplementedError("Model type not implemented")
            
            pred = torch.argmax(logits, dim=1)
            true_labels.extend(label.cpu().numpy())
            predictions.extend(pred.cpu().numpy())

    # Calculate overall accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Overall Accuracy: {accuracy:.3f}")

    # Calculate precision, recall, and F1 score for each class and display them
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average=None)
    classes = sorted(set(true_labels))
    for i, class_label in enumerate(classes):
        print(f"Class {class_label} - Precision: {precision[i]:.3f}, Recall: {recall[i]:.3f}, F1 Score: {f1[i]:.3f}")

# NOTE: you will also have to go into the respective config file and change the model_type to the model_type you want to run inference on