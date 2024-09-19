import numpy as np
import random
import torch

SEED = 42


def set_seed(seed: int = None):
    # Set random seed
    if seed is None:
        seed = SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
