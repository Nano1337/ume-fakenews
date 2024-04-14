
# Basic Libraries
import os 
import argparse
import yaml

# Deep Learning Libraries
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything 
from torch.utils.data import DataLoader

# internal files
from fakeddit.get_data import get_data, get_sampler
from fakeddit import get_model
from utils.run_trainer import run_trainer
from utils.setup_configs import setup_configs

# set reproducible 
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('medium')

def run_training():
    """ 
    Data:
    - batch[0] is (B, S) text, modality x2
    - batch[1] is (B, C, H, W) image, modality x1
    - batch[2] is [B] labels, depends on num_classes
    Optionally: 
    - batch[3] is (B) idx, for qmf model
    """

    args = setup_configs()

    train_dataset, val_dataset, test_dataset = get_data(args)

    # get dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_cpus, 
        persistent_workers=True,
        prefetch_factor = 4,
        collate_fn=train_dataset.collate_fn,
        sampler=get_sampler(train_dataset), 
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_cpus, 
        persistent_workers=True, 
        prefetch_factor=4,
        shuffle=False, 
        collate_fn=val_dataset.collate_fn, 
        pin_memory=True, 
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_cpus, 
        persistent_workers=True, 
        prefetch_factor=4,
        shuffle=False,
        collate_fn=test_dataset.collate_fn, 
        pin_memory=True,
    )

    # get model
    model = get_model(args)

    run_trainer(args, model, train_loader, val_loader, test_loader)