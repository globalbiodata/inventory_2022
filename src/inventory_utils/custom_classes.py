"""
Custom Classes
~~~

Custom classes for the Biodata Resource Inventory.

Authors: Kenneth Schackart
"""

import argparse
import sys
from typing import Any, NamedTuple

import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW


# ---------------------------------------------------------------------------
class CustomHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """ Custom Argparse help formatter """
    def _get_help_string(self, action):
        """ Suppress defaults that are None """
        if action.default is None:
            return action.help
        return super()._get_help_string(action)

    def _format_action_invocation(self, action):
        """ Show metavar only once """
        if not action.option_strings:
            metavar, = self._metavar_formatter(action, action.dest)(1)
            return metavar
        parts = []
        if action.nargs == 0:
            parts.extend(action.option_strings)
        else:
            default = action.dest.upper()
            args_string = self._format_args(action, default)
            for option_string in action.option_strings:
                parts.append('%s' % option_string)
            parts[-1] += ' %s' % args_string
        return ', '.join(parts)


# ---------------------------------------------------------------------------
class Splits(NamedTuple):
    """
    Training, validation, and test dataframes

    `train`: Training data
    `val`: Validation data
    `test`: Test data
    """
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


# ---------------------------------------------------------------------------
class Settings(NamedTuple):
    """
    Settings used for model training

    `model`: Pretrained model
    `optimizer`: Training optimizer
    `train_dataloader`: `DataLoader` of training data
    `val_dataloader`: `DataLoader` of validation data
    `lr_scheduler`: Learning rate schedule (optional)
    `num_epochs`: Maximum number of training epochs
    `num_training_steps`: Maximum number of training steps
    (`num_epochs` * `num_training`)
    `device`: Torch device
    """

    model: Any
    optimizer: AdamW
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    lr_scheduler: Any
    num_epochs: int
    num_training_steps: int
    device: torch.device


# ---------------------------------------------------------------------------
class Metrics(NamedTuple):
    """
    Performance metrics

    `precision`: Model precision
    `recall`: Model recall
    `f1`: Model F1 score
    `loss`: Model loss
    """

    precision: float
    recall: float
    f1: float
    loss: float


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    sys.exit('This file is a module, and is not meant to be run.')
