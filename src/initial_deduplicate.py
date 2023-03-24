#!/usr/bin/env python3
"""
Purpose: Deduplicate rows with identical URL and name
Authors: Kenneth Schackart
"""

import argparse
import os
import re
from typing import NamedTuple, TextIO

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from process_names import wrangle_names
from inventory_utils.custom_classes import CustomHelpFormatter
from inventory_utils.wrangling import join_commas

pd.options.mode.chained_assignment = None


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    file: TextIO
    previous: TextIO
    out_dir: str


# ---------------------------------------------------------------------------
def get_args() -> Args:
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description=('Deduplicate rows with identical URL and name'),
        formatter_class=CustomHelpFormatter)

    parser.add_argument('file',
                        metavar='FILE',
                        type=argparse.FileType('rt', encoding='ISO-8859-1'),
                        help='CSV file of predictions and metadata')
    parser.add_argument('-p',
                        '--previous',
                        metavar='FILE',
                        type=argparse.FileType('rt', encoding='ISO-8859-1'),
                        help='Previously processed inventory')
    parser.add_argument('-o',
                        '--out-dir',
                        metavar='DIR',
                        type=str,
                        default='out/',
                        help='Output directory')

    args = parser.parse_args()

    return Args(args.file, args.previous, args.out_dir)


# ---------------------------------------------------------------------------
@pytest.fixture(name='raw_data')
def fixture_raw_data() -> pd.DataFrame:
    """ DataFrame representative of the input data """

    columns = [
        'ID', 'text', 'common_name', 'common_prob', 'full_name', 'full_prob',
        'extracted_url', 'best_common', 'best_common_prob', 'best_full',
        'best_full_prob', 'best_name', 'best_name_prob', 'publication_date'
    ]

    df = pd.DataFrame(
        [[
            '123', 'The text', 'mmCIF, PDB', '0.987, 0.775',
            'Protein Data Bank', '0.717', 'http://www.pdb.org/', 'mmCIF',
            '0.987', 'Protein Data Bank', '0.717', 'mmCIF', '0.987',
            '2011-01-01'
        ],
         [
             '456', 'More text.', '', '', 'SBASE', '0.648',
             'http://www.icgeb.trieste.it/sbase', '', '', 'SBASE', '0.648',
             'SBASE', '0.648', '2011-01-02'
         ],
         [
             '147', 'Wow.', 'TwoURLS', '0.998', '', '',
             'http://website.com, http://db.org', 'TwoURLS', '0.998', '', '',
             'TwoURLS', '0.998', '2011-01-03'
         ],
         [
             '369', 'The cat drank wine', 'PDB', '0.963', 'Protein Data Bank',
             '0.964', 'http://www.pdb.org/', 'PDB', '0.963',
             'Protein Data Bank', '0.964', 'Protein Data Bank', '0.964',
             '2011-01-04'
         ],
         [
             '741', 'Almost 7eleven', 'PDB', '0.983', 'Protein Data Bank',
             '0.964', 'http://www.pdb.org/', 'PDB', '0.983',
             'Protein Data Bank', '0.964', 'PDB', '0.983', '2011-01-05'
         ],
         [
             '852', 'Chihiro', 'PDB', '0.963', 'Protein Data Bank', '0.984',
             'http://www.pdb.org/', 'PDB', '0.963', 'Protein Data Bank',
             '0.984', 'Protein Data Bank', '0.984', '2011-01-06'
         ]],
        columns=columns)

    return df


# ---------------------------------------------------------------------------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataframe for potential merging of old results deduplication

    Parameters:
    `df`: Input dataframe

    Return: Cleaned dataframe
    """

    df = df.drop(['common_name', 'common_prob', 'full_name', 'full_prob'],
                 axis='columns')
    all_columns = df.columns
    df[all_columns] = df[all_columns].fillna('').astype(str)
    df['extracted_url'] = df['extracted_url'].map(clean_url)

    return df


# ---------------------------------------------------------------------------
def prep_previous(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare previous inventory for merging with new results

    Parameters:
    `df`: Previous inventory results

    Return: Dataframe with same columns as new results
    """

    df['text'] = ''

    columns = [
        'ID', 'text', 'extracted_url', 'best_common', 'best_common_prob',
        'best_full', 'best_full_prob', 'best_name', 'best_name_prob',
        'publication_date'
    ]

    df = df[columns]

    return df


# ---------------------------------------------------------------------------
def test_prep_previous(raw_data: pd.DataFrame) -> None:
    """ Test prep_previous() """

    in_df = clean_df(raw_data)

    new_columns = in_df.columns

    # Previous results are already deduplicated
    previous = deduplicate(raw_data)

    # Add extra columns to simulate previously obtained results
    previous['extracted_url_status'] = '400'
    previous['extracted_url_country'] = 'USA'

    previous = prep_previous(previous)

    prev_columns = previous.columns

    assert all(new_columns == prev_columns)


# ---------------------------------------------------------------------------
def integrate_previous(new_df: pd.DataFrame,
                       prev_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add previous results so that all can be deduplicated

    Parameters:
    `new_df`: New data
    `prev_df`: Previously processed inventory

    Return: Combined dataframe
    """

    prev_df = prep_previous(prev_df)
    out_df = pd.concat([new_df, prev_df])

    return out_df


# ---------------------------------------------------------------------------
def test_integrate_previous(raw_data: pd.DataFrame) -> None:
    """ Test integrate_previous() """

    previous = deduplicate(raw_data)

    # Add extra columns to simulate previously obtained results
    previous['extracted_url_status'] = '400'
    previous['extracted_url_country'] = 'USA'

    out_df = integrate_previous(raw_data, previous)

    assert len(out_df) == 11


# ---------------------------------------------------------------------------
def clean_url(url: str) -> str:
    """
    For the sake of matching URLs, remove trailing slash, replace
    https:// with http://, and lowercase all before first single slash

    Parameters:
    `url`: URL string

    Return: Cleaned URL
    """

    # Split at first single slash to lowercase the first half
    url_parts = re.search(
        r'''(?P<before_slash>.*?) # Group everything before first slash
        (?<!/)                    # No preceding slash
        /                         # Match single slash
        (?!/)                     # No following slash
        (?P<after_slash>.*)       # Group everything after first slash
        ''', url, re.X)

    if url_parts:
        url = url_parts['before_slash'].lower(
        ) + '/' + url_parts['after_slash']
    else:
        url = url.lower()

    return re.sub('https', 'http', url.rstrip('/'))


# ---------------------------------------------------------------------------
def test_clean_url() -> None:
    """ Test clean_url() """

    # Does not modify good URL
    assert clean_url('http://mirdb.org') == 'http://mirdb.org'

    # Removes trailing slashes
    assert clean_url('http://mirdb.org/') == 'http://mirdb.org'

    # Replaces https with http
    assert clean_url('https://mirdb.org') == 'http://mirdb.org'

    # Does both
    assert clean_url('https://mirdb.org/') == 'http://mirdb.org'

    # Lowercases domain
    assert clean_url('http://mycoCLAP.fungalgenomics.ca'
                     ) == 'http://mycoclap.fungalgenomics.ca'

    # Does not lowercase anything after first single slash
    assert clean_url('http://MYDB.com/BASE') == 'http://mydb.com/BASE'


# ---------------------------------------------------------------------------
def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate the resource dataframe by finding resources with the same
    names.

    Parameters:
    `df`: Dataframe that has gone through URL filtering and name wrangling
    `thresh`: Threshold probability for which a name can be used for
    deduplication by matching
    `common`: Use common name for matching, if above `thresh`
    `full`: Use full name for matching, if above `thresh`
    `url`: Use URLs for matching

    Return: Deduplicated dataframe
    """

    duplicates = df.duplicated(['best_name', 'extracted_url'], keep=False)

    unique_df = df[~duplicates]
    duplicate_df = df[duplicates]
    duplicate_df['article_count'] = 0

    duplicate_df = (duplicate_df.sort_values(
        'publication_date',
        ascending=False).groupby(['best_name', 'extracted_url']).agg({
            'ID':
            join_commas,
            'text':
            'first',
            'best_common':
            join_commas,
            'best_common_prob':
            join_commas,
            'best_full':
            join_commas,
            'best_full_prob':
            join_commas,
            'article_count':
            len,
            'publication_date':
            'first'
        }).reset_index())

    unique_df['article_count'] = 1

    if len(duplicate_df) > 0:
        duplicate_df = wrangle_names(duplicate_df, 'best_common',
                                     'best_common_prob', 'best_full',
                                     'best_full_prob')

    out_df = pd.concat([unique_df, duplicate_df])

    return out_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
def test_deduplicate(raw_data: pd.DataFrame) -> None:
    """ Test deduplicate() """

    out_ids = pd.Series(['123', '456', '147', '741', '852, 369'], name='ID')
    out_citations = pd.Series([1, 1, 1, 1, 2], name='article_count')

    return_df = deduplicate(raw_data)

    assert_series_equal(return_df['ID'], out_ids)
    assert_series_equal(return_df['article_count'], out_citations)


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

    in_df = clean_df(pd.read_csv(args.file, dtype=str))

    if args.previous:
        in_df = integrate_previous(in_df, pd.read_csv(args.previous,
                                                      dtype=str))

    out_df = deduplicate(in_df)

    outfile = make_filename(out_dir, args.file.name)

    out_df.to_csv(outfile, index=False)

    print(f'Done. Wrote output to {outfile}.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
