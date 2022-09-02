#!/usr/bin/env python3
"""
Purpose: Finalize inventory by deduplicating and filtering
Authors: Kenneth Schackart
"""

import argparse
import os
from itertools import chain
from typing import Dict, Iterator, List, NamedTuple, TextIO, Union

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from inventory_utils.custom_classes import CustomHelpFormatter

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
def get_args() -> Args:
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description=('Finalize the inventory:'
                     ' filter out bad predictions,'
                     ' deduplicate,'
                     ' aggregate,'
                     ' summarize'),
        formatter_class=CustomHelpFormatter)

    parser.add_argument('file',
                        metavar='FILE',
                        type=argparse.FileType('rt', encoding='ISO-8859-1'),
                        help='CSV file of predictions and metadata')
    parser.add_argument('-o',
                        '--out-dir',
                        metavar='DIR',
                        type=str,
                        default='out/',
                        help='Output directory')
    parser.add_argument('-nu',
                        '--min-urls',
                        metavar='INT',
                        type=int,
                        default=1,
                        help=('Minimum number of URLs per resource.'
                              ' Resources with less discarded.'))
    parser.add_argument('-xu',
                        '--max-urls',
                        metavar='INT',
                        type=int,
                        default=2,
                        help=('Maximum number of URLs per resource.'
                              ' Resources with more are discarded.'
                              ' (0 = No maximum)'))
    parser.add_argument('-np',
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
        'extracted_url', 'extracted_url_status', 'extracted_url_country',
        'extracted_url_coordinates'
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
                max_urls: int) -> pd.DataFrame:
    """
    Filter dataframe to only include rows with
    `min_urls` <= URLs <= `max_urls`.

    Parameters:
    `df`: Raw dataframe
    `min_urls`: Minimum number of URLS per row/article (0=no min)
    `max_urls`: Maximum number of URLs per row/article (0=no max)

    Return: Filtered dataframe
    """
    df['url_count'] = df['extracted_url'].map(lambda x: len(x.split(', ')))

    no_urls = df['extracted_url'].values == ''
    df['url_count'][no_urls] = 0

    enough_urls = min_urls <= df['url_count'].values
    out_df = df[enough_urls]

    if max_urls:
        not_too_many_urls = out_df['url_count'].values <= max_urls
        out_df = out_df[not_too_many_urls]

    out_df = out_df.drop(['url_count'], axis=1)

    return out_df


# ---------------------------------------------------------------------------
def test_filter_urls(raw_data: pd.DataFrame) -> None:
    """ Test filter_urls() """
    original_row_count = len(raw_data)
    original_cols = raw_data.columns

    # Doesn't remove any rows; no min and no max
    out_df = filter_urls(raw_data, 0, 0)
    assert len(out_df) == original_row_count
    assert (out_df.columns == original_cols).all()

    # Removes row with 4 URLs; no min
    out_df = filter_urls(raw_data, 0, 3)
    assert len(out_df) == original_row_count - 1
    assert (out_df.columns == original_cols).all()

    # Removes row with no URLs; no max
    out_df = filter_urls(raw_data, 1, 0)
    assert len(out_df) == original_row_count - 1
    assert (out_df.columns == original_cols).all()

    # Removes row with 0 and 4 URLs
    out_df = filter_urls(raw_data, 1, 3)
    assert len(out_df) == original_row_count - 2
    assert (out_df.columns == original_cols).all()


# ---------------------------------------------------------------------------
def make_dict(keys: List, values: Union[List, Iterator[float]]) -> Dict:
    """
    Make a dictionary from lists of keys and values

    Parameters:
    `keys`: list of keys
    `values`: list of values

    Return: Dictionary
    """

    return dict(zip(keys, values))


# ---------------------------------------------------------------------------
def test_make_dict() -> None:
    """ Test make_dict() """

    names = ['mmCIF', 'PDB']
    probs = [0.987, 0.775]

    assert make_dict(names, probs) == {'mmCIF': 0.987, 'PDB': 0.775}


# ---------------------------------------------------------------------------
def concat_dicts(*args: Dict) -> Dict:
    """
    Concatenate multiple dictionaries into one

    Parameters:
    `*args`: Any number of dictionaries

    Return: Concatenated dictionary
    """

    return dict(chain.from_iterable(d.items() for d in args))


# ---------------------------------------------------------------------------
def test_combine_dicts() -> None:
    """ Test combine_dicts() """

    comm = {'mmCIF': 0.987, 'PDB': 0.775}
    full = {'Protein Data Bank': 0.717}
    combined = {'mmCIF': 0.987, 'PDB': 0.775, 'Protein Data Bank': 0.717}

    assert concat_dicts(comm, full) == combined


# ---------------------------------------------------------------------------
def select_names(common_names: str, common_probs: str, full_names: str,
                 full_probs: str) -> pd.Series:
    """
    Select common name with highest probability, full name with highest
    probability, and name with overall highest probability

    Parameters:
    `common_names`: Predicted common name(s)
    `common_probs`: Probabilities of predicted common name(s)
    `full_names`: Predicted full name(s)
    `full_probs`: Probabilities of predicted full name(s)

    Return: Pandas Series withprobable common name, probable full name,
    best overall name, probability of best overall name
    """
    def convert_number(s: str) -> float:
        return float(s) if s else 0

    common_dict = make_dict(common_names.split(', '),
                            map(convert_number, common_probs.split(', ')))
    full_dict = make_dict(full_names.split(', '),
                          map(convert_number, full_probs.split(', ')))
    combined_dict = concat_dicts(full_dict, common_dict)

    best_common = sorted(
        common_dict,
        key=common_dict.get,  # type: ignore
        reverse=True)[0]
    best_full = sorted(
        full_dict,
        key=full_dict.get,  # type: ignore
        reverse=True)[0]
    best_name = sorted(
        combined_dict,
        key=combined_dict.get,  # type: ignore
        reverse=True)[0]
    best_prob = combined_dict[best_name]

    return pd.Series(
        [best_common, best_full, best_name, best_prob],
        index=['best_common', 'best_full', 'best_name', 'best_name_prob'])


# ---------------------------------------------------------------------------
def test_select_names() -> None:
    """ Test select_names() """

    idx = ['best_common', 'best_full', 'best_name', 'best_name_prob']
    # Only one found
    in_list = ['LBD2000', '0.997', '', '']
    output = pd.Series(['LBD2000', '', 'LBD2000', 0.997], index=idx)
    assert_series_equal(select_names(*in_list), output)

    # Common name is better
    in_list = ['PDB', '0.983', 'Protein Data Bank', '0.964']
    output = pd.Series(['PDB', 'Protein Data Bank', 'PDB', 0.983], index=idx)
    assert_series_equal(select_names(*in_list), output)

    # Full name is better
    in_list = ['PDB', '0.963', 'Protein Data Bank', '0.984']
    output = pd.Series(
        ['PDB', 'Protein Data Bank', 'Protein Data Bank', 0.984], index=idx)
    assert_series_equal(select_names(*in_list), output)

    # Multiple to unpack
    in_list = ['mmCIF, PDB', '0.987, 0.775', 'Protein Data Bank', '0.717']
    output = pd.Series(['mmCIF', 'Protein Data Bank', 'mmCIF', 0.987],
                       index=idx)
    assert_series_equal(select_names(*in_list), output)

    # Equal probability, favor full name
    in_list = ['PDB', '0.963', 'Protein Data Bank', '0.963']
    output = pd.Series(
        ['PDB', 'Protein Data Bank', 'Protein Data Bank', 0.963], index=idx)
    assert_series_equal(select_names(*in_list), output)


# ---------------------------------------------------------------------------
def wrangle_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Place best common name, best full name, best overall name, and best name
    probability in new columns

    Parameters:
    `df`: Dataframe

    Return: Dataframe with 4 new columns
    """

    new_cols = ['best_common', 'best_full', 'best_name', 'best_name_prob']

    df[new_cols] = df.apply(lambda x: list(
        select_names(x['common_name'], x['common_prob'], x['full_name'], x[
            'full_prob'])),
                            axis=1,
                            result_type='expand')

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
def test_wrangle_names(raw_data: pd.DataFrame) -> None:
    """ Test wrangle_names() """

    in_df = filter_urls(raw_data, 0, 0)

    out_df = wrangle_names(in_df)

    best_common = pd.Series([
        'mmCIF', '', 'LDB2000', 'TwoURLS', 'LotsaURLS', 'PDB', 'PDB', 'PDB', ''
    ],
                            name='best_common')
    best_full = pd.Series([
        'Protein Data Bank', 'SBASE', '', '', '', 'Protein Data Bank',
        'Protein Data Bank', 'Protein Data Bank', ''
    ],
                          name='best_full')
    best_overall = pd.Series([
        'mmCIF', 'SBASE', 'LDB2000', 'TwoURLS', 'LotsaURLS',
        'Protein Data Bank', 'PDB', 'Protein Data Bank', ''
    ],
                             name='best_name')
    best_prob = pd.Series(
        [0.987, 0.648, 0.997, 0.998, 0.996, 0.964, 0.983, 0.984, 0],
        name='best_name_prob')

    assert_series_equal(out_df['best_common'], best_common)
    assert_series_equal(out_df['best_full'], best_full)
    assert_series_equal(out_df['best_name'], best_overall)
    assert_series_equal(out_df['best_name_prob'], best_prob)


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

    print(f'Done. Wrote output to {outfile}.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
