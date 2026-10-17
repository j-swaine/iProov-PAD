"""
Seed for full reproducibility of experiments.
Inspired by:
* https://docs.pytorch.org/docs/2.12/notes/randomness.html
* https://medium.com/data-science/random-seeds-and-reproducibility-933da79446e3
* https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
"""

from __future__ import annotations

import os
import random


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Generate a global seed for all libraries to ensure result reproducibility."""

    # hash randomisation is normally used.
    # here we set the PYTHONHASHSEED environment variable to a fixed value to ensure that the hash values of objects are reproducible across different runs of the program.
    # each library maintains its own random state, so we need to seed manually.
    # Read once at process startup by CPython -- this means it won't alter the seed of the parent process once running if called late but it can influence child processes.
    os.environ["PYTHONHASHSEED"] = str(seed)
    # seed the native random module to ensure that any random operations performed using this module are reproducible.
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)  # seed the cpu
        # seed all gpus (covers DDP multi-gpu)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass
    return seed
