"""
Runtime
~~~

Functions that modify or detect the runtime.

Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import sys

import torch


# ---------------------------------------------------------------------------
def set_random_seed(seed: int):
    """
    Set random seed for deterministic outcome of ML-trained models

    `seed`: Value to use for seed
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
def get_torch_device() -> torch.device:
    """
    Get device for torch

    Return:
    `torch.device` either "cuda" or "cpu"
    """

    return torch.device('cuda') if torch.cuda.is_available() else torch.device(
        'cpu')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    sys.exit('This file is a module, and is not meant to be run.')
