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
    out_dir: str


# ---------------------------------------------------------------------------
def get_args() -> Args:
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Choose model based on highest validation F1 score',
        formatter_class=CustomHelpFormatter)

    parser.add_argument('files',
                        nargs='+',
                        metavar='FILE',
                        type=str,
                        help='Training stat files of models to be compared')
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

    return Args(args.files, args.out_dir)


# ---------------------------------------------------------------------------
def get_model_name(filename: str) -> str:
    """
    Get model name from input file name

    Parameters:
    `filename`: Input filename containing model name
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
    """

    return os.path.split(filename)[0]


# ---------------------------------------------------------------------------
def test_get_checkpoint_dir() -> None:
    """ Test get_checkpoint_dir """

    chkpt_dir = 'out/classif_checkpoints/scibert'
    _, filename = make_filenames(chkpt_dir)

    assert get_checkpoint_dir(filename) == chkpt_dir


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    out_df = pd.DataFrame()
    best_f1 = 0.
    best_model = ''
    for filename in args.files:
        model_name = get_model_name(filename)
        checkpoint_dir = get_checkpoint_dir(filename)
        df = pd.read_csv(filename)

        if max(df['val_f1']) > best_f1:
            best_model, _ = make_filenames(checkpoint_dir)

        df['model'] = model_name
        out_df = pd.concat([out_df, df])

    out_df.reset_index(inplace=True, drop=True)

    out_dir = os.path.join(args.out_dir, model_name)

    model_outfile = os.path.join(out_dir, 'best_checkpt.pt')
    stats_outfile = os.path.join(out_dir, 'combined_stats.csv')

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_df.to_csv(stats_outfile, index=False)
    shutil.copyfileobj(open(best_model, 'rb'), open(model_outfile, 'wb'))

    print(f'Checkpoint of best model is {best_model}')
    print('Done. Wrote combined stats file and best model checkpoint',
          f'to {out_dir}')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
