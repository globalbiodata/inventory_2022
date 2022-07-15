#!/usr/bin/env python3
"""
Purpose: Choose model based on highest metric of choice
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
    """
    name: str
    checkpt: str


# ---------------------------------------------------------------------------
def get_args() -> Args:
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Choose model with highest validation metric of choice',
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
def get_best_model(df: pd.DataFrame, metric: str) -> BestModel:
    """
    Determine best model from the training stats

    Parameters:
    `df`: Training stats dataframe
    `metric`: Metric to use as criteria

    Return: Best model attributes
    """

    best_epoch = df.sort_values(by=[metric, 'val_loss'],
                                ascending=[False, True]).iloc[0]

    best_model = BestModel(best_epoch['model'], best_epoch['checkpt'])

    return best_model


# ---------------------------------------------------------------------------
def test_get_best_model() -> None:
    """ Test get_best_model() """

    cols = [
        'epoch', 'train_precision', 'train_recall', 'train_f1', 'train_loss',
        'val_precision', 'val_recall', 'val_f1', 'val_loss', 'checkpt', 'model'
    ]

    in_df = pd.DataFrame([[
        3, 1., 1., 1., 0., 0.926, 0.820, 0.920, 0.00803, 'scibert.pt',
        'scibert'
    ],
                          [
                              5, 1., 1., 1., 0., 0.926, 0.820, 0.870, 0.00703,
                              'biobert.pt', 'biobert'
                          ]],
                         columns=cols)

    # scibert has higher validation f1
    assert get_best_model(in_df,
                          'val_f1') == BestModel('scibert', 'scibert.pt')

    # They have same precision, but biobert has lower val_loss on that epoch
    assert get_best_model(in_df, 'val_precision') == BestModel(
        'biobert', 'biobert.pt')


# ---------------------------------------------------------------------------
def write_outputs(best_model: BestModel, out_df: pd.DataFrame,
                  out_dir: str) -> str:
    """
    Copy best model checkpoint to {out_dir}/{model_name} and write combined
    training stats csv file

    Parameters:
    `best_model`: Attributes of best model
    `out_df`: Combined training stats dataframe
    `out_dir`: Output directory

    Return: Full output directory
    """

    out_dir = os.path.join(out_dir, best_model.name)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    model_outfile = os.path.join(out_dir, 'best_checkpt.pt')
    stats_outfile = os.path.join(out_dir, 'combined_stats.csv')

    out_df.drop(['checkpt'], axis='columns', inplace=True)
    out_df.to_csv(stats_outfile, index=False)
    shutil.copyfileobj(open(best_model.checkpt, 'rb'),
                       open(model_outfile, 'wb'))

    return out_dir


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()

    out_df = pd.DataFrame()
    for filename in args.files:
        df = pd.read_csv(filename)

        checkpt, _ = make_filenames(get_checkpoint_dir(filename))
        df['checkpt'] = checkpt
        df['model'] = get_model_name(filename)
        out_df = pd.concat([out_df, df])

    best_model = get_best_model(out_df, 'val_' + args.metric)

    out_dir = write_outputs(best_model, out_df, args.out_dir)

    print(f'Checkpoint of best model is {best_model.checkpt}')
    print('Done. Wrote combined stats file and best model checkpoint',
          f'to {out_dir}')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
