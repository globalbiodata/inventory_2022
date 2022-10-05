#!/usr/bin/env python3
"""
Purpose: Flag rows for manual review
Authors: Kenneth Schackart
"""

import argparse
import os
from itertools import chain
from typing import Dict, Iterator, List, NamedTuple, TextIO, Tuple, Union

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
    min_urls: int
    max_urls: int
    min_prob: float


# ---------------------------------------------------------------------------
class FilterResults(NamedTuple):
    """
    Result of filtering

    `df`: Filtered `DataFrame`
    `under_urls`: Number of rows with too few URLs
    `over_urls`: Number of rows with too many URLs
    `no_names`: Number of rows with no predicted names
    """
    df: pd.DataFrame
    under_urls: int
    over_urls: int
    no_names: int


# ---------------------------------------------------------------------------
def get_args() -> Args:
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(description=('Flag rows for review '
                                                  'based on number of URLs, '
                                                  'name probability, and '
                                                  'possible duplication.'),
                                     formatter_class=CustomHelpFormatter)

    inputs = parser.add_argument_group('Inputs and Outputs')
    filters = parser.add_argument_group('Parameters for Filtering')

    inputs.add_argument('file',
                        metavar='FILE',
                        type=argparse.FileType('rt', encoding='ISO-8859-1'),
                        help='CSV file of predictions and metadata')
    inputs.add_argument('-o',
                        '--out-dir',
                        metavar='DIR',
                        type=str,
                        default='out/',
                        help='Output directory')
    filters.add_argument('-nu',
                         '--min-urls',
                         metavar='INT',
                         type=int,
                         default=1,
                         help=('Minimum number of URLs per resource.'
                               ' Resources with less discarded.'))
    filters.add_argument('-u',
                         '--max-urls',
                         metavar='INT',
                         type=int,
                         default=2,
                         help=('Maximum number of URLs per resource.'
                               ' Resources with more are flagged.'
                               ' (0 = No maximum)'))
    filters.add_argument(
        '-p',
        '--min_prob',
        metavar='PROB',
        type=float,
        default=0.95,
        help=('Minimum probability of predicted resource name.'
              ' Anything below will be flagged for review.'))

    args = parser.parse_args()

    if args.min_urls < 0:
        parser.error(f'--min-urls cannot be less than 0; got {args.min_urls}')

    return Args(args.file, args.out_dir, args.min_urls, args.max_urls,
                args.min_prob)


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
def filter_urls(df: pd.DataFrame, min_urls: int,
                max_urls: int) -> Tuple[pd.DataFrame, int, int]:
    """
    Filter dataframe to only include rows with
    `min_urls` <= URLs <= `max_urls`.

    Parameters:
    `df`: Raw dataframe
    `min_urls`: Minimum number of URLS per row/article (0=no min)
    `max_urls`: Maximum number of URLs per row/article (0=no max)

    Return: Filtered dataframe, number of no URLs, number of too
    many URLs
    """

    df = df.copy()

    df['url_count'] = df['extracted_url'].map(lambda x: len(x.split(', ')))

    no_urls = df['extracted_url'].values == ''
    df['url_count'][no_urls] = 0

    enough_urls = min_urls <= df['url_count'].values
    under_urls = sum([val is np.bool_(False) for val in list(enough_urls)])
    out_df = df[enough_urls]

    over_urls = 0
    if max_urls:
        not_too_many_urls = out_df['url_count'].values <= max_urls
        over_urls = sum([val is np.bool_(False) for val in not_too_many_urls])
        out_df = out_df[not_too_many_urls]

    out_df.drop(['url_count'], axis='columns', inplace=True)

    return out_df, under_urls, over_urls


# ---------------------------------------------------------------------------
def test_filter_urls(raw_data: pd.DataFrame) -> None:
    """ Test filter_urls() """
    original_row_count = len(raw_data)
    original_cols = raw_data.columns

    # Doesn't remove any rows; no min and no max
    out_df, _, _ = filter_urls(raw_data, 0, 0)
    assert len(out_df) == original_row_count
    assert (out_df.columns == original_cols).all()

    # Removes row with 4 URLs; no min
    out_df, _, _ = filter_urls(raw_data, 0, 3)
    assert len(out_df) == original_row_count - 1
    assert (out_df.columns == original_cols).all()

    # Removes row with no URLs; no max
    out_df, _, _ = filter_urls(raw_data, 1, 0)
    assert len(out_df) == original_row_count - 1
    assert (out_df.columns == original_cols).all()

    # Removes row with 0 and 4 URLs
    out_df, _, _ = filter_urls(raw_data, 1, 3)
    assert len(out_df) == original_row_count - 2
    assert (out_df.columns == original_cols).all()


# ---------------------------------------------------------------------------
def filter_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove articles for which no names were predicted

    Parameters:
    `df`: Input dataframe

    Return: Dataframe with rows without names are removed
    """

    return df[df['best_name'] != '']


# ---------------------------------------------------------------------------
def test_filter_names(raw_data: pd.DataFrame) -> None:
    """ Test filter_names() """

    filt_df, _, _ = filter_urls(raw_data, 0, 0)
    in_df = flag_for_review(wrangle_names(filt_df), 0.99)

    # Article ID 963 is the only article without any predicted names
    remaining_article_ids = pd.Series(
        ['123', '456', '789', '147', '258', '369', '741', '852'], name='ID')

    return_df = filter_names(in_df)

    assert (in_df.columns == return_df.columns).all()
    assert_series_equal(return_df['ID'], remaining_article_ids)


# ---------------------------------------------------------------------------
def filter_df(df: pd.DataFrame, min_urls: int, max_urls: int,
              min_prob: float) -> FilterResults:
    """
    Filter dataframe based on URLs and names

    Parameters:
    `df`: Input dataframe
    # `min_urls`: Minimum number of URLs
    `max_urls`: Maximum number of URLs
    `min_prob`: Minimum probability of best name, flag those below

    Return: `FilterResults` object with dataframe and filter stats
    """

    orig_rows = len(df)

    url_filt_df, under_urls, over_urls = filter_urls(df, min_urls, max_urls)

    name_filt_df = filter_names(df)
    num_bad_names = orig_rows - len(name_filt_df)

    # out_df = pd.merge(url_filt_df, name_filt_df, how='inner', on='ID')
    out_df = pd.merge(url_filt_df, name_filt_df, how='inner')

    return FilterResults(out_df, under_urls, over_urls, num_bad_names)


# ---------------------------------------------------------------------------
def test_filter_df(raw_data: pd.DataFrame) -> None:
    """ Test filter_df() """

    filt_results = filter_df(raw_data, 1, 2, 0.9)

    assert filt_results.under_urls == 1
    assert filt_results.over_urls == 1
    assert filt_results.no_names == 1
    assert filt_results.review == 1


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
    for id, value in zip(ids, values):
        matches = []
        for split_value in value.split(','):
            match_mask = [
                split_value in other_value.split(',') for other_value in values
            ]
            id_matches = ids[match_mask]
            id_matches = [match for match in id_matches if match != id]
            matches += id_matches

        out.append(join_commas(matches))

    print(out)

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

    in_df = pd.read_csv(args.file).fillna('').drop_duplicates(['ID'])
    orig_rows = len(in_df)

    filt_results = filter_df(
        args.min_urls,
        args.max_urls,
    )
    out_df = flag_df(filt_results.df, args.min_prob)

    end_rows = len(filt_results.df)

    print(f'Done filtering results:\n'
          f'\tOriginal Number of articles: {orig_rows}\n'
          f'\t- Too few URLs: {filt_results.under_urls}\n'
          f'\t- Too many URLs: {filt_results.over_urls}\n'
          f'\t- Articles with no name: {filt_results.no_names}\n'
          f'\tNumber remaining articles: {end_rows}\n'
          f'\tNumber marked for review: {filt_results.review}')

    filt_results.df.to_csv(outfile, index=False)

    print(f'Wrote output to {outfile}.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
