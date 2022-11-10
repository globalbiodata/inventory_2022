#!/usr/bin/env python3
"""
Purpose: Flag rows for manual review
Authors: Kenneth Schackart
"""

import argparse
import os
from typing import NamedTuple, TextIO

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from inventory_utils.custom_classes import CustomHelpFormatter
from inventory_utils.wrangling import join_commas

pd.options.mode.chained_assignment = None


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    file: TextIO
    out_dir: str
    min_prob: float


# ---------------------------------------------------------------------------
class FlaggingStats(NamedTuple):
    """
    Counts of flagged rows

    `total_flags`: Total number of flagged rows
    `duplicate_urls`: Number of rows flagged for duplicate URLs
    `duplicate_names`: Number of rows flagged for duplicate names
    `low_probs`: Number of rows flagged for low probability name
    """
    total_flags: int
    duplicate_urls: int
    duplicate_names: int
    low_probs: int


# ---------------------------------------------------------------------------
def get_args() -> Args:
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(description=('Flag rows for review '
                                                  'based on best '
                                                  'name probability, and '
                                                  'possible duplication.'),
                                     formatter_class=CustomHelpFormatter)

    parser.add_argument('file',
                        metavar='FILE',
                        type=argparse.FileType('rt', encoding='ISO-8859-1'),
                        help='CSV file of articles')
    parser.add_argument('-o',
                        '--out-dir',
                        metavar='DIR',
                        type=str,
                        default='out/',
                        help='Output directory')
    parser.add_argument('-p',
                        '--min_prob',
                        metavar='PROB',
                        type=float,
                        default=0.95,
                        help=('Minimum probability of predicted resource name.'
                              ' Anything below will be flagged for review.'))

    args = parser.parse_args()

    return Args(args.file, args.out_dir, args.min_prob)


# ---------------------------------------------------------------------------
@pytest.fixture(name='raw_data')
def fixture_raw_data() -> pd.DataFrame:
    """ DataFrame representative of the input data """

    columns = [
        'ID', 'text', 'common_name', 'common_prob', 'full_name', 'full_prob',
        'extracted_url', 'best_common', 'best_common_prob', 'best_full',
        'best_full_prob', 'best_name', 'best_name_prob'
    ]

    df = pd.DataFrame(
        [
            [  # Two common names, one full
                '123', 'The text', 'mmCIF, PDB', '0.987, 0.775',
                'Protein Data Bank', '0.717', 'http://www.pdb.org/', '200',
                'US', '(34.22,-118.24)'
            ],
            [  # No common name, low probability full name
                '456', 'More text.', '', '', 'SBASE', '0.648',
                'http://www.icgeb.trieste.it/sbase', '301', '', ''
            ],
            [  # No URL
                '789', 'Stuff.', 'LDB2000', '0.997', '', '', '', '', '', ''
            ],
            [  # Two URLS
                '147', 'Wow.', 'TwoURLS', '0.998', '', '',
                'http://website.com, http://db.org', '200, 302', 'JP, GB',
                '(35.67,139.65), (52.20,0.13)'
            ],
            [  # Many URLs
                '258', 'Sawasdee', 'LotsaURLS', '0.996', '', '',
                'http://db.com, http://site.org, http://res.net, http://db.io',
                '404, Exception, 200, 301', ', , JP, GB',
                ', , (35.67,139.65), (52.20,0.13)'
            ],
            [  # Same name as 123, but not highest prob
                '369', 'The cat drank wine', 'PDB', '0.963',
                'Protein Data Bank', '0.964', 'http://www.pdb.org/', '200',
                'US', '(34.22,-118.24)'
            ],
            [  # Shared highest prob name with 369
                '741', 'Almost 7eleven', 'PDB', '0.983', 'Protein Data Bank',
                '0.964', 'http://www.pdb.org/', '200', 'US', '(34.22,-118.24)'
            ],
            [  # Same common and full names, mismatched prob ranking with 741
                '852', 'Chihiro', 'PDB', '0.963', 'Protein Data Bank', '0.984',
                'http://www.pdb.org/', '200', 'US', '(34.22,-118.24)'
            ],
            [  # No names predicted
                '963', 'Sen', '', '', '', '', 'http://www.pdb.org/', '200',
                'US', '(34.22,-118.24)'
            ]
        ],
        columns=columns)

    return df


# ---------------------------------------------------------------------------
def flag_duplicates(ids: pd.Series, values: pd.Series) -> pd.Series:
    """
    Create column which indicates potential duplicates based on the given
    column. New column values are ID's of that row's potential duplicate

    Parameters:
    `ids`: Column of IDs
    `values`: Column of values which may have duplicates

    Return: Columns with potential duplicate IDs
    """

    out = []
    for id_n, value in zip(ids, values):
        matches = []
        for split_value in value.split(','):
            match_mask = [
                split_value in other_value.split(',') for other_value in values
            ]
            id_matches = ids[match_mask]
            id_matches = [match for match in id_matches if match != id_n]
            matches += id_matches

        out.append(join_commas(matches))

    return pd.Series(out)


# ---------------------------------------------------------------------------
def test_flag_duplicates() -> None:
    """ Test flag_duplicates() """

    ids = pd.Series(['123', '456', '789', '147', '258', '369'])

    names = pd.Series(['name1', 'name2', 'name3', 'name1', 'name4', 'name1'])
    expected_flagged_names = pd.Series(
        ['147, 369', '', '', '123, 369', '', '123, 147'])
    flagged_names = flag_duplicates(ids, names)
    assert_series_equal(flagged_names, expected_flagged_names)

    urls = pd.Series(
        ['url1', 'url2', 'url1', 'url13, url4', 'url2', 'url1, url5'])
    expected_flagged_urls = pd.Series(
        ['789, 369', '258', '123, 369', '', '456', '123, 789'])
    flagged_urls = flag_duplicates(ids, urls)
    assert_series_equal(flagged_urls, expected_flagged_urls)


# ---------------------------------------------------------------------------
def flag_probs(probs: pd.Series, min_prob: float) -> pd.Series:
    """
    Flag rows with probability below `min_prob`

    Parameters:
    `probs`: Column of Probabilities
    `min_prob`: Minimum probability for flagging

    Return: Column of strings with flagged rows
    """

    probs = probs.astype(float)

    out_col = np.where(probs < min_prob, 'low_prob_best_name', '')

    return pd.Series(out_col)


# ---------------------------------------------------------------------------
def test_flag_probs() -> None:
    """ Test flag_probs() """

    min_prob = 0.6
    in_col = pd.Series(['0.00', '0.50', '0.75', '1.00'])
    out_col = pd.Series(['low_prob_best_name', 'low_prob_best_name', '', ''])

    assert_series_equal(flag_probs(in_col, min_prob), out_col)


# ---------------------------------------------------------------------------
def flag_df(df: pd.DataFrame, min_prob: float):
    """
    Flag dataframe for manual review. Two columns are added:
    potential_duplicates, and check_out

    Parameters:
    `df`: Input dataframe
    `min_prob`: Minimum probability of best name, flag those below

    Return:
    """

    df['duplicate_urls'] = flag_duplicates(df['ID'], df['extracted_url'])
    df['duplicate_names'] = flag_duplicates(df['ID'], df['best_name'])

    df['low_prob'] = flag_probs(df['best_name_prob'], min_prob)

    return df


# ---------------------------------------------------------------------------
def count_flags(url_flags: pd.Series, name_flags: pd.Series,
                prob_flags: pd.Series) -> FlaggingStats:
    """
    Count the number of rows that have been flagged for manual review

    Parameters:
    `df`: Flagged dataframe

    Return: Number of flagged rows
    """

    num_url_flags = sum(url_flags != '')
    num_name_flags = sum(name_flags != '')
    num_prob_flags = sum(prob_flags != '')

    any_flags = [(url_flag == name_flag == prob_flag == '')
                 for url_flag, name_flag, prob_flag in zip(
                     url_flags, name_flags, prob_flags)]
    num_any_flag = any_flags.count(False)

    return FlaggingStats(num_any_flag, num_url_flags, num_name_flags,
                         num_prob_flags)


# ---------------------------------------------------------------------------
def test_count_flags() -> None:
    """ Test count_flags() """

    url_flags = pd.Series(['', '258', '', '', '456', ''])
    name_flags = pd.Series(['', '', '147, 258', '789, 258', '789, 147', ''])
    prob_flags = pd.Series(
        ['low_prob_best_name', 'low_prob_best_name', '', '', '', ''])

    expected_counts = FlaggingStats(5, 2, 3, 2)

    assert count_flags(url_flags, name_flags, prob_flags) == expected_counts


# ---------------------------------------------------------------------------
def add_review_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns that are to be filled during manual review

    Parameters:
    `df`: Flagged dataframe

    Return: Dataframe with new (empty)columns
    """

    df[[
        'review_low_prob', 'review_dup_urls', 'review_dup_names',
        'review_notes_low_prob', 'review_notes_dup_url',
        'review_notes_dup_names'
    ]] = ''

    return df


# ---------------------------------------------------------------------------
def make_filename(out_dir: str, infile_name: str) -> str:
    '''
    Make filename for output reusing input file's basename

    Parameters:
    `out_dir`: Output directory
    `infile_name`: Input file name

    Return: Output filename
    '''

    return os.path.join(out_dir, os.path.basename(infile_name))


# ---------------------------------------------------------------------------
def test_make_filenames() -> None:
    """ Test make_filenames() """

    assert make_filename(
        'out/checked_urls',
        'out/urls/predictions.csv') == ('out/checked_urls/predictions.csv')


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()
    out_dir = args.out_dir

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    outfile = make_filename(out_dir, args.file.name)

    in_df = pd.read_csv(args.file,
                        dtype=str).fillna('').drop_duplicates(['ID'])

    flagged_df = flag_df(in_df, args.min_prob)

    num_flagged = count_flags(flagged_df['duplicate_urls'],
                              flagged_df['duplicate_names'],
                              flagged_df['low_prob'])

    print(f'Total number of flagged rows: {num_flagged.total_flags}')
    print(f'Rows with duplicate names: {num_flagged.duplicate_names}')
    print(f'Rows with duplicate URLs: {num_flagged.duplicate_urls}')
    print(f'Rows with low probability name: {num_flagged.low_probs}')

    out_df = add_review_columns(flagged_df)

    out_df.to_csv(outfile, index=False)

    print(f'Wrote output to {outfile}.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
