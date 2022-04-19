#!/usr/bin/env python3
"""
Purpose: Split curated data into training, validation, and testing sets
Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import argparse
import os
import pytest
import sys
from typing import List, NamedTuple, TextIO

import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.model_selection import train_test_split

from utils import CustomHelpFormatter


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """
    Command-line arguments

    `infile`: Input curated data filehandle
    `outdir`: Output directory
    `train`: Training data output file name
    `val`: Validation data output file name
    `spltis`: Train, val, test proportions
    `test`: Test data output file name
    `seed`: Random seed
    """
    infile: TextIO
    outdir: str
    train: str
    val: str
    test: str
    splits: List[float]
    seed: bool


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
@pytest.fixture(name='unsplit_data')
def fixture_unsplit_data() -> pd.DataFrame:
    """ Example dataframe for testing splitting function """

    df = pd.DataFrame([[123, 'First title', 'First abstract', 0],
                       [456, 'Second title', 'Second abstract', 1],
                       [789, 'Third title', 'Third abstract', 0],
                       [321, 'Fourth title', 'Fourth abstract', 1],
                       [654, 'Fifth title', 'Fifth abstract', 0],
                       [987, 'Sixth title', 'Sixth abstract', 1],
                       [741, 'Seventh title', 'Seventh abstract', 0],
                       [852, 'Eighth title', 'Eighth abstract', 1]],
                      columns=['id', 'title', 'abstract', 'curation_score'])

    return df


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
                        '--outdir',
                        metavar='',
                        type=str,
                        default='data/',
                        help='Output directory')
    parser.add_argument('-t',
                        '--train',
                        metavar='',
                        type=str,
                        default='train_paper_classif.csv',
                        help='Training data output file name')
    parser.add_argument('-v',
                        '--val',
                        metavar='',
                        type=str,
                        default='val_paper_classif.csv',
                        help='Validation data output file name')
    parser.add_argument('-s',
                        '--test',
                        metavar='',
                        type=str,
                        default='test_paper_classif.csv',
                        help='Test data output file name')
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

    return Args(args.infile, args.outdir, args.train, args.val, args.test,
                args.splits, args.seed)


# ---------------------------------------------------------------------------
def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only data with curation score of 0 or 1

    `df`: Manually curated data
    """

    df = df[['id', 'title', 'abstract', 'curation_score']]

    return df[df['curation_score'].isin([0, 1])]


# ---------------------------------------------------------------------------
def test_filter_data() -> None:
    """ Test filter_data() """

    in_df = pd.DataFrame(
        [[123, 'First title', 'First abstract', 0, 'nope'],
         [456, 'Second title', 'Second abstract', 1, 'yup'],
         [789, 'Third title', 'Third abstract', 0.5, 'unsure']],
        columns=['id', 'title', 'abstract', 'curation_score', 'notes'])

    out_df = pd.DataFrame(
        [[123, 'First title', 'First abstract', 0],
         [456, 'Second title', 'Second abstract', 1]],
        columns=['id', 'title', 'abstract', 'curation_score'])

    assert_frame_equal(filter_data(in_df), out_df, check_dtype=False)


# ---------------------------------------------------------------------------
def check_data(df: pd.DataFrame) -> None:
    """ Check that input data is valid """

    num_certain = df['id'].count()
    unique_ids = df['id'].nunique()

    if not num_certain == unique_ids:
        sys.exit(f'Number of certain scores ({num_certain}) not equal to'
                 f'number of unique IDs ({unique_ids}).')


# ---------------------------------------------------------------------------
def split_df(df: pd.DataFrame, rand_seed: bool, splits: List[float]) -> Splits:
    """
    Split manually curated data into train, validation and test sets
    
    `df`: Manually curated classification data
    `seed`: Optionally use random seed
    """

    seed = 241 if rand_seed else None

    _, val_split, test_split = splits
    val_test_split = val_split + test_split

    train, val_test = train_test_split(df,
                                       test_size=val_test_split,
                                       random_state=seed)
    val, test = train_test_split(val_test,
                                 test_size=test_split / val_test_split,
                                 random_state=seed)

    return Splits(train, val, test)


# ---------------------------------------------------------------------------
def test_random_split(unsplit_data: pd.DataFrame) -> None:
    """ Test that split_df() gives correct proportions """

    in_df = unsplit_data

    train, val, test = split_df(in_df, False, [0.5, 0.25, 0.25])

    assert len(train.index) == 4
    assert len(val.index) == 2
    assert len(test.index) == 2


# ---------------------------------------------------------------------------
def test_seeded_split(unsplit_data: pd.DataFrame) -> None:
    """ Test that split_df() behaves deterministically """

    in_df = unsplit_data

    train, val, test = split_df(in_df, True, [0.5, 0.25, 0.25])

    assert list(train['id'].values) == [321, 789, 741, 654]
    assert list(val['id'].values) == [987, 456]
    assert list(test['id'].values) == [852, 123]


# ---------------------------------------------------------------------------
def make_filename(dir: str, name: str) -> str:
    """
    Make output filename

    `dir`: Output directory
    `name`: File name
    """

    outfile = os.path.join(dir, name)

    return outfile


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()
    out_dir = args.outdir

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    df = pd.read_csv(args.infile)

    df = filter_data(df)

    check_data(df)

    train_df, val_df, test_df = split_df(df, args.seed, args.splits)

    train_out, val_out, test_out = map(lambda f: make_filename(out_dir, f),
                                       [args.train, args.val, args.test])

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)

    print(f'Done. Wrote 3 files to {out_dir}.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
