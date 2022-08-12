#!/usr/bin/env python3
"""
Purpose: Choose model based on highest metric of choice
Authors: Kenneth Schackart
"""

import argparse
import os
from typing import BinaryIO, Dict, List, NamedTuple, Union, cast

import pandas as pd
import torch

from inventory_utils.custom_classes import CustomHelpFormatter, Metrics


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    checkpoints: List[BinaryIO]
    metric: str
    out_dir: str


# ---------------------------------------------------------------------------
def get_args() -> Args:
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Choose model with highest validation metric of choice',
        formatter_class=CustomHelpFormatter)

    parser.add_argument('checkpointss',
                        nargs='+',
                        metavar='FILE',
                        type=argparse.FileType('rb'),
                        help='Model checkpoints to be compared')
    parser.add_argument('-m',
                        '--metric',
                        metavar='METRIC',
                        choices=['f1', 'precision', 'recall'],
                        default='f1',
                        type=str,
                        help='Metric to use for choosing best model')
    parser.add_argument('-o',
                        '--out-dir',
                        metavar='DIR',
                        type=str,
                        default='out/',
                        help='Output directory')

    args = parser.parse_args()

    return Args(args.checkpoints, args.metric, args.out_dir)


# ---------------------------------------------------------------------------
def get_metrics(checkpoint_fh: BinaryIO) -> Dict[str, Union[float, str]]:
    """
    Retrieve the validation metrics from model checkpoint

    Parameters:
    `checkpoint_fh`: Trained model checkpoint

    Return: Dictionary of validation set metrics
    """

    checkpoint = torch.load(checkpoint_fh)
    metrics = cast(Metrics, checkpoint['val_metrics'])

    return {
        'f1': metrics.f1,
        'precision': metrics.precision,
        'recall': metrics.recall,
        'loss': metrics.loss
    }


# ---------------------------------------------------------------------------
def get_best_model(df: pd.DataFrame, metric: str) -> str:
    """
    Determine best model from the training stats

    Parameters:
    `df`: Chosen models dataframe
    `metric`: Metric to use as criteria

    Return: Best model checkpoint
    """

    best_model = df.sort_values(by=[metric, 'loss'], ascending=[False,
                                                                True]).iloc[0]

    return best_model['checkpt']


# ---------------------------------------------------------------------------
def test_get_best_model() -> None:
    """ Test get_best_model() """

    cols = ['f1', 'precision', 'recall', 'loss', 'checkpt']

    in_df = pd.DataFrame([[
        0.92,
        0.926,
        0.82,
        0.008,
        'scibert.pt',
    ], [0.87, 0.926, 0.82, 0.007, 'biobert.pt']],
                         columns=cols)

    # scibert has higher f1
    assert get_best_model(in_df, 'f1') == 'scibert.pt'

    # They have same precision, but biobert has lower loss on that epoch
    assert get_best_model(in_df, 'precision') == 'biobert.pt'


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    all_metrics = pd.DataFrame()
    for checkpoint in args.checkpoints:
        metrics = get_metrics(checkpoint)
        metrics['checkpt'] = checkpoint.name
        all_metrics = pd.concat([all_metrics, metrics])

    best_model = get_best_model(all_metrics, args.metric)

    out_file = os.path.join(args.out_dir, 'best_checkpt.txt')
    with open(out_file) as out_fh:
        print(best_model, file=out_fh)

    print(f'Best model checkpoint is {best_model}.')
    print(f'Done. Wrote output to {out_file}.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
