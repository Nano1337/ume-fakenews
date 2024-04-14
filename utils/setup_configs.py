import argparse
import os
from pytorch_lightning import seed_everything
from utils.merge_yaml import load_and_merge_yaml


def setup_configs(): 
    """
    Purpose: 
        - Parse command line arguments 
        - Load and merge yaml files 
        - Set reproducibility
    Args: 
        None
    Returns:
        args: object with parsed arguments
    """

    # load configs into args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None) 
    args = parser.parse_args()
    cfg = None
    if args.dir:
        override_file = os.path.join(args.dir, args.dir + ".yaml")
        base_file = os.path.join("utils", "base_cfg.yaml")
        cfg = load_and_merge_yaml(base_file, override_file)
    else:
        raise NotImplementedError("No directory provided, please specify flag --dir")
    for key, val in cfg.items():
        setattr(args, key, val)

    seed_everything(args.seed, workers=True) 

    return args