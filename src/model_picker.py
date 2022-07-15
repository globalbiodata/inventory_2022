#!/usr/bin/env python3
"""
Purpose: Choose model based on highest validation F1 score
Authors: Kenneth Schackart
"""

import argparse
import os
import shutil
from typing import List, NamedTuple

import pandas as pd

from utils import CustomHelpFormatter, make_filenames


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    files: List[str]
    metric: str
    out_dir: str


# ---------------------------------------------------------------------------
class BestModel(NamedTuple):
    """
    Attributes of best model

    `model_name`: Name of model
    `model_checkpt`: Checkpoint location
    `metric`: Value of metric used as criteria
    `loss`: Loss value
    """
    name: str
    checkpt: str
    metric: float
    loss: float


# ---------------------------------------------------------------------------
def get_args() -> Args:
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Choose model based on highest validation metric',
        formatter_class=CustomHelpFormatter)

    parser.add_argument('files',
                        nargs='+',
                        metavar='FILE',
                        type=str,
                        help='Training stat files of models to be compared')
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

    for file in args.files:
        if not os.path.isfile(file):
            parser.error(f'Input file "{file}" does not exist.')

    return Args(args.files, args.metric, args.out_dir)


# ---------------------------------------------------------------------------
def get_model_name(filename: str) -> str:
    """
    Get model name from input file name

    e.g. 'out/scibert/checkpoint.pt' -> 'scibert'

    Parameters:
    `filename`: Input filename containing model name

    Return: Model (short)name
    """

    model_name = filename.split('/')[-2]

    return model_name


# ---------------------------------------------------------------------------
def test_get_model_name() -> None:
    """ Test get_model_name """

    model_name = 'scibert'
    _, filename = make_filenames('out/' + model_name)
    assert get_model_name(filename) == model_name

    model_name = 'biomed_roberta'
    _, filename = make_filenames('out/' + model_name)
    assert get_model_name(filename) == model_name


# ---------------------------------------------------------------------------
def get_checkpoint_dir(filename: str) -> str:
    """
    Get checkpoint directory from input file name

    Parameters:
    `filename`: Input filename

    Return:
    Checkpoint directory name
    """

    return os.path.split(filename)[0]


# ---------------------------------------------------------------------------
def test_get_checkpoint_dir() -> None:
    """ Test get_checkpoint_dir """

    chkpt_dir = 'out/classif_checkpoints/scibert'
    _, filename = make_filenames(chkpt_dir)

    assert get_checkpoint_dir(filename) == chkpt_dir


# ---------------------------------------------------------------------------
def compare_metrics(filename: str, df: pd.DataFrame, current_best: BestModel,
                    metric: str) -> BestModel:
    """
    Compare training stats in df to current best model

    Parameters:
    `filename`: Current training stats file
    `df`: Training stats dataframe
    `current_best`: Attributes of current best model
    `metric`: Metric to use as criteria

    Return: Updated best model attributes
    """

    best_epoch = df.sort_values(by=[metric, 'val_loss'],
                                ascending=False).iloc[0]
    better_metric = best_epoch[metric] > current_best.metric
    equal_metric = best_epoch[metric] == current_best.metric
    better_loss = best_epoch['val_loss'] < current_best.loss

    if better_metric or (equal_metric and better_loss):
        checkpt, _ = make_filenames(get_checkpoint_dir(filename))
        current_best = BestModel(get_model_name(filename), checkpt,
                                 best_epoch[metric], best_epoch['val_loss'])

    return current_best


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()

    metric = 'val_' + args.metric

    out_df = pd.DataFrame()
    best_model = BestModel('', '', 0., 1.)
    for filename in args.files:
        model_name = get_model_name(filename)

        df = pd.read_csv(filename)

        best_model = compare_metrics(filename, df, best_model, metric)

        df['model'] = model_name
        out_df = pd.concat([out_df, df])

    out_df.reset_index(inplace=True, drop=True)

    out_dir = os.path.join(args.out_dir, best_model.name)

    model_outfile = os.path.join(out_dir, 'best_checkpt.pt')
    stats_outfile = os.path.join(out_dir, 'combined_stats.csv')

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_df.to_csv(stats_outfile, index=False)
    shutil.copyfileobj(open(best_model.checkpt, 'rb'),
                       open(model_outfile, 'wb'))

    print(f'Checkpoint of best model is {best_model.checkpt}')
    print('Done. Wrote combined stats file and best model checkpoint',
          f'to {out_dir}')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
