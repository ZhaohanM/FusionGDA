import os
import random
import time

import numpy as np
import torch


def set_random_seed(seed):
    """set random seeds.

    Args:
        seed (int): random seed number.
    """
    # Set random seed for reproducibility
    if seed is None:
        seed = int(time.time())
        print(f"Generated random seed {seed}.")
    else:
        print(f"Received random seed {seed}.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def save_model(args, total_step, model):
    if args.use_adapter:
        adapter_save_path = save_dir = os.path.join(
            args.save_path_prefix,
            f"reduction_factor_{args.reduction_factor}_lr_{args.lr}",
        )
        model.save_adapters(adapter_save_path, total_step)
        print(f"save model adapters on {args.save_path_prefix}")
    else:
        save_dir = os.path.join(
            args.save_path_prefix,
            f"step_{total_step}_model.bin",
        )
        torch.save(model, save_dir)
        print(f"save model {save_dir} on {args.save_path_prefix}")
