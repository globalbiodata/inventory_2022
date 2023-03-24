#!/usr/bin/env python3
"""
Purpose: Split curated data into training, validation, and testing sets
Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import argparse
import os
import sys
from typing import List, NamedTuple, TextIO

import pandas as pd
from pandas.testing import assert_frame_equal

from inventory_utils.custom_classes import CustomHelpFormatter
from inventory_utils.wrangling import split_df


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """
    Command-line arguments

    `infile`: Input curated data filehandle
    `outdir`: Output directory
    `splits`: Train, val, test proportions
    `seed`: Random seed
    """
    infile: TextIO
    outdir: str
    splits: List[float]
    seed: bool


# ---------------------------------------------------------------------------
def get_args() -> Args:
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Split curated classification data',
        formatter_class=CustomHelpFormatter)

    parser.add_argument('infile',
                        metavar='FILE',
                        type=argparse.FileType('rt', encoding='ISO-8859-1'),
                        default='data/manual_classifications.csv',
                        help='Manually classified input file')
    parser.add_argument('-o',
                        '--out-dir',
                        metavar='',
                        type=str,
                        default='data/classif_splits',
                        help='Output directory')
    parser.add_argument('--splits',
                        metavar='',
                        type=float,
                        nargs=3,
                        default=[0.7, 0.15, 0.15],
                        help='Proportions for train, val, test splits')
    parser.add_argument('-r',
                        '--seed',
                        action='store_true',
                        help='Set random seed')

    args = parser.parse_args()

    if not sum(args.splits) == 1.0:
        parser.error(f'--splits {args.splits} must sum to 1')

    return Args(args.infile, args.out_dir, args.splits, args.seed)


# ---------------------------------------------------------------------------
def check_input(df: pd.DataFrame) -> None:
    """
    Check the input data columns

    Parameters:
    `df`: Input dataframe
    """

    exp_cols = ['id', 'title', 'abstract', 'curation_score']

    if not all(col in df.columns for col in exp_cols):
        sys.exit(
            f'ERROR: Input data does not have the expected columns: {exp_cols}'
        )


# ---------------------------------------------------------------------------
def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only data with curation score of 0 or 1

    Parameters:
    `df`: Manually curated data

    Return: Filtered dataframe
    """

    df = df[['id', 'title', 'abstract', 'curation_score']]

    return df[df['curation_score'].isin(['0', '1'])]


# ---------------------------------------------------------------------------
def test_filter_data() -> None:
    """ Test filter_data() """

    in_df = pd.DataFrame(
        [[123, 'First title', 'First abstract', '0', 'nope'],
         [456, 'Second title', 'Second abstract', '1', 'yup'],
         [789, 'Third title', 'Third abstract', '0.5', 'unsure']],
        columns=['id', 'title', 'abstract', 'curation_score', 'notes'])

    out_df = pd.DataFrame(
        [[123, 'First title', 'First abstract', '0'],
         [456, 'Second title', 'Second abstract', '1']],
        columns=['id', 'title', 'abstract', 'curation_score'])

    assert_frame_equal(filter_data(in_df), out_df, check_dtype=False)


# ---------------------------------------------------------------------------
def check_data(df: pd.DataFrame) -> None:
    """
    Check that input data is valid, with same numnber of curation scores as
    number of unique id's

    Parameters:
    `df`: Curated data
    """

    num_certain = df['id'].count()
    unique_ids = df['id'].nunique()

    if not num_certain == unique_ids:
        sys.exit(f'ERROR: Number of certain scores ({num_certain}) not equal'
                 f' to number of unique IDs ({unique_ids}).')


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()
    out_dir = args.outdir

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    df = pd.read_csv(args.infile, dtype=str)

    check_input(df)

    df = filter_data(df)

    check_data(df)

    train_df, val_df, test_df = split_df(df, args.seed, args.splits)

    train_out, val_out, test_out = map(lambda f: os.path.join(out_dir, f), [
        'train_paper_classif.csv', 'val_paper_classif.csv',
        'test_paper_classif.csv'
    ])

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)

    print(f'Done. Wrote 3 files to {out_dir}.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
