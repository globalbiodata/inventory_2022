#!/usr/bin/env python3
"""
Purpose: Finalize inventory by deduplicating and filtering
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
    `review`: Number of rows flagged for review
    """
    df: pd.DataFrame
    under_urls: int
    over_urls: int
    no_names: int
    review: int


# ---------------------------------------------------------------------------
def get_args() -> Args:
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(description=('Filter the inventory'),
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
    filters.add_argument('-xu',
                         '--max-urls',
                         metavar='INT',
                         type=int,
                         default=2,
                         help=('Maximum number of URLs per resource.'
                               ' Resources with more are discarded.'
                               ' (0 = No maximum)'))
    filters.add_argument(
        '-np',
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
def make_dict(keys: List, values: Union[List, Iterator[float]]) -> Dict:
    """
    Make a dictionary from lists of keys and values

    Parameters:
    `keys`: list of keys
    `values`: list of values

    Return: Dictionary
    """

    return dict([(key, value) if len(key) != 1 else ('', 0)
                 for key, value in zip(keys, values)])


# ---------------------------------------------------------------------------
def test_make_dict() -> None:
    """ Test make_dict() """

    names = ['mmCIF', 'PDB', 'A']
    probs = [0.987, 0.775, 0.95]

    assert make_dict(names, probs) == {'mmCIF': 0.987, 'PDB': 0.775, '': 0}


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

    Return: Pandas Series with probable common name, probable full name,
    best overall name, and probabilities of each
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
    best_common_prob = combined_dict[best_common]
    best_full = sorted(
        full_dict,
        key=full_dict.get,  # type: ignore
        reverse=True)[0]
    best_full_prob = combined_dict[best_full]
    best_name = sorted(
        combined_dict,
        key=combined_dict.get,  # type: ignore
        reverse=True)[0]
    best_prob = combined_dict[best_name]

    return pd.Series([
        best_common, best_common_prob, best_full, best_full_prob, best_name,
        best_prob
    ],
                     index=[
                         'best_common', 'best_common_prob', 'best_full',
                         'best_full_prob', 'best_name', 'best_name_prob'
                     ])


# ---------------------------------------------------------------------------
def test_select_names() -> None:
    """ Test select_names() """

    idx = [
        'best_common', 'best_common_prob', 'best_full', 'best_full_prob',
        'best_name', 'best_name_prob'
    ]
    # Only one found
    in_list = ['LBD2000', '0.997', '', '']
    output = pd.Series(['LBD2000', 0.997, '', 0, 'LBD2000', 0.997], index=idx)
    assert_series_equal(select_names(*in_list), output)

    # Common name is better
    in_list = ['PDB', '0.983', 'Protein Data Bank', '0.964']
    output = pd.Series(
        ['PDB', 0.983, 'Protein Data Bank', 0.964, 'PDB', 0.983], index=idx)
    assert_series_equal(select_names(*in_list), output)

    # Full name is better
    in_list = ['PDB', '0.963', 'Protein Data Bank', '0.984']
    output = pd.Series(
        ['PDB', 0.963, 'Protein Data Bank', 0.984, 'Protein Data Bank', 0.984],
        index=idx)
    assert_series_equal(select_names(*in_list), output)

    # Multiple to unpack
    in_list = ['mmCIF, PDB', '0.987, 0.775', 'Protein Data Bank', '0.717']
    output = pd.Series(
        ['mmCIF', 0.987, 'Protein Data Bank', 0.717, 'mmCIF', 0.987],
        index=idx)
    assert_series_equal(select_names(*in_list), output)

    # Equal probability, favor full name
    in_list = ['PDB', '0.963', 'Protein Data Bank', '0.963']
    output = pd.Series(
        ['PDB', 0.963, 'Protein Data Bank', 0.963, 'Protein Data Bank', 0.963],
        index=idx)
    assert_series_equal(select_names(*in_list), output)

    # Single letter name
    in_list = ['mmCIF, A', '0.987, 0.99', 'F', '0.717']
    output = pd.Series(['mmCIF', 0.987, '', 0, 'mmCIF', 0.987], index=idx)
    assert_series_equal(select_names(*in_list), output)


# ---------------------------------------------------------------------------
def wrangle_names(df: pd.DataFrame,
                  common_col: str = 'common_name',
                  common_prob_col: str = 'common_prob',
                  full_name_col: str = 'full_name',
                  full_prob_col: str = 'full_prob') -> pd.DataFrame:
    """
    Place best common name, best full name, best overall name, and best name
    probability in new columns

    Parameters:
    `df`: Dataframe

    Return: Dataframe with 4 new columns
    """

    new_cols = [
        'best_common', 'best_common_prob', 'best_full', 'best_full_prob',
        'best_name', 'best_name_prob'
    ]

    df[new_cols] = df.apply(lambda x: list(
        select_names(x[common_col], x[common_prob_col], x[full_name_col], x[
            full_prob_col])),
                            axis=1,
                            result_type='expand')

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
def test_wrangle_names(raw_data: pd.DataFrame) -> None:
    """ Test wrangle_names() """

    in_df, _, _ = filter_urls(raw_data, 0, 0)

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
def flag_for_review(df: pd.DataFrame, thresh: float) -> pd.DataFrame:
    """
    Flag rows with best name probability < `thresh` for manual review

    Parameters:
    `df`: Dataframe with wrangled names
    `thresh`: Threshold for flagging

    Return: Flagged dataframe
    """

    df['confidence'] = ''
    df['confidence'][df['best_name_prob'] < thresh] = 'manual_review'

    return df


# ---------------------------------------------------------------------------
def test_flag_for_review(raw_data: pd.DataFrame) -> None:
    """ Test flag_for_review() """

    filt_df, _, _ = filter_urls(raw_data, 0, 0)
    in_df = wrangle_names(filt_df)

    thresh = 0.99
    confidence = pd.Series([
        'manual_review', 'manual_review', '', '', '', 'manual_review',
        'manual_review', 'manual_review', 'manual_review'
    ],
                           name='confidence')

    out_df = flag_for_review(in_df, thresh)

    assert_series_equal(out_df['confidence'], confidence)


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
    `min_urls`: Minimum number of URLs
    `max_urls`: Maximum number of URLs
    `min_prob`: Minimum probability of best name, flag those below

    Return: `FilterResults` object with dataframe and filter stats
    """

    orig_rows = len(df)

    url_filt_df, under_urls, over_urls = filter_urls(df, min_urls, max_urls)

    name_filt_df = filter_names(flag_for_review(wrangle_names(df), min_prob))
    num_bad_names = orig_rows - len(name_filt_df)

    num_review = sum(name_filt_df['confidence'] == 'manual_review')

    out_df = pd.merge(url_filt_df, name_filt_df, how='inner', on='ID')

    return FilterResults(out_df, under_urls, over_urls, num_bad_names,
                         num_review)


# ---------------------------------------------------------------------------
def test_filter_df(raw_data: pd.DataFrame) -> None:
    """ Test filter_df() """

    filt_results = filter_df(raw_data, 1, 2, 0.9)

    assert filt_results.under_urls == 1
    assert filt_results.over_urls == 1
    assert filt_results.no_names == 1
    assert filt_results.review == 1


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

    filt_results = filter_df(in_df, args.min_urls, args.max_urls,
                             args.min_prob)

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
