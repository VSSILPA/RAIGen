import torch
from training import train_sae
from sae import GlobalBatchTopKMatryoshkaSAE
from config import get_default_cfg, post_init_cfg
from datasets import Dataset
import os
import random
import numpy as np
from raigen import identify_minority

import argparse

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def parse_args():
    default_cfg = get_default_cfg()
    parser = argparse.ArgumentParser()

    for key, value in default_cfg.items():
        if isinstance(value, bool):
            # Handle boolean flags
            parser.add_argument(f'--{key}', action='store_true' if not value else 'store_false', default=value)
        elif isinstance(value, list):
            # Handle lists
            parser.add_argument(f'--{key}', type=type(value[0]), nargs='+', default=value)
        elif isinstance(value, torch.dtype):
            # Convert dtype to string (e.g., 'float32')
            parser.add_argument(f'--{key}', type=str, default=str(value).replace('torch.', ''))
        else:
            parser.add_argument(f'--{key}', type=type(value), default=value)

    args = parser.parse_args()
    # Convert back dtype string to torch.dtype if necessary
    if hasattr(args, 'dtype'):
        args.dtype = getattr(torch, args.dtype)
    return args

if __name__ == "__main__":
    cfg = parse_args()
    cfg = post_init_cfg(cfg)
    set_seed(seed=cfg.seed)
    dataset_dict = {}
    dtype = cfg.dtype

    dataset = Dataset.load_from_disk(
                        os.path.join(cfg.dataset_path, cfg.hook_point), keep_in_memory=False
                    )
    dataset.set_format(
        type="torch",
        columns=["activations", "timestep"],
        dtype=dtype,
    )

    dataset = dataset.shuffle(cfg.seed)
    dataset_dict[cfg.hook_point] = dataset

    sae = GlobalBatchTopKMatryoshkaSAE(cfg)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("torch.cuda.device_count:", torch.cuda.device_count())
    print("cfg.device:", cfg.device)

    print("SAE param device:", next(sae.parameters()).device)
    print("SAE param dtype  :", next(sae.parameters()).dtype)

            
    if cfg.train_sae:
        train_sae(sae, dataset_dict, cfg)
    save_dir =  f"{cfg.checkpoint_root}/{cfg.name}/final"

    state_dict = torch.load(save_dir + '/sae.pt', map_location="cpu")
    sae.load_state_dict(state_dict)
    sae.training = False
    sae = sae.half()
    identify_minority(cfg, save_dir, sae)

