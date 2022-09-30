#!/usr/bin/env python3
"""
Purpose: Process predicted names
Authors: Kenneth Schackart
"""

import argparse
import os
from itertools import chain
from typing import Dict, Iterator, List, NamedTuple, TextIO, Tuple, Union

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


# ---------------------------------------------------------------------------
def get_args() -> Args:
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(description=('Process predicted names'),
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

    args = parser.parse_args()

    return Args(args.file, args.out_dir)


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
def make_dict(keys: List, values: Union[List, Iterator[float]]) -> Dict:
    """
    Make a dictionary from lists of keys and values

    Parameters:
    `keys`: list of keys
    `values`: list of values

    Return: Dictionary
    """

    # Replace single character keys (names) with empty string
    keys = [key if len(key) != 1 else '' for key in keys]

    # Assign zero probability (value) to empty strings
    return {key: value if key != '' else 0 for key, value in zip(keys, values)}


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
    in_list = ['mmCIF, A', '0.987, 0.99', 'F, G', '0.717, 0.912']
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

    out_df = wrangle_names(raw_data)

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

    in_df = wrangle_names(raw_data)

    # Article ID 963 is the only article without any predicted names
    remaining_article_ids = pd.Series(
        ['123', '456', '789', '147', '258', '369', '741', '852'], name='ID')

    return_df = filter_names(in_df)

    assert (in_df.columns == return_df.columns).all()
    assert_series_equal(return_df['ID'], remaining_article_ids)


# ---------------------------------------------------------------------------
def filter_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Determine best short and full names, and remove rows with no names

    Parameters:
    `df`: Input dataframe

    Return: Tuple of Dataframe, number of no-names
    """

    orig_rows = len(df)

    out_df = filter_names(wrangle_names(df))
    num_bad_names = orig_rows - len(out_df)

    return out_df, num_bad_names


# ---------------------------------------------------------------------------
def test_filter_df(raw_data: pd.DataFrame) -> None:
    """ Test filter_df() """

    _, num_no_name = filter_df(raw_data)

    assert num_no_name == 1


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

    out_df, num_no_name = filter_df(in_df)

    plu = 's' if num_no_name != 1 else ''
    print(f'Done processing names.\n{num_no_name} '
          f'article{plu} with no names removed.')

    out_df.to_csv(outfile, index=False)

    print(f'Wrote output to {outfile}.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
