#!/usr/bin/env python3
"""
Purpose: Finalize inventory by deduplicating and filtering
Authors: Kenneth Schackart
"""

import argparse
import os
from typing import NamedTuple, TextIO

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from inventory_utils.custom_classes import CustomHelpFormatter
from inventory_utils.wrangling import join_commas
from filter_results import wrangle_names

pd.options.mode.chained_assignment = None


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    file: TextIO
    out_dir: str
    common: bool
    full: bool
    url: bool


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

    inputs = parser.add_argument_group('Inputs and Outputs')
    filters = parser.add_argument_group('Parameters for Filtering')
    dedupe = parser.add_argument_group('Fields to Match for Deduplication '
                                       '(not mutually-exclusive)')

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

    dedupe.add_argument('--match-common',
                        help='Match on common name, even if not best name',
                        action='store_true')
    dedupe.add_argument('--match-full',
                        help='Match on full name, even if not best name',
                        action='store_true')
    dedupe.add_argument('--match-url',
                        help='Match on URL',
                        action='store_true')

    args = parser.parse_args()

    if args.min_urls < 0:
        parser.error(f'--min-urls cannot be less than 0; got {args.min_urls}')

    return Args(args.file, args.out_dir, args.match_common, args.match_full,
                args.match_url)


# ---------------------------------------------------------------------------
@pytest.fixture(name='raw_data')
def fixture_raw_data() -> pd.DataFrame:
    """ DataFrame representative of the input data """

    columns = [
        'ID', 'text', 'common_name', 'common_prob', 'full_name', 'full_prob',
        'extracted_url', 'extracted_url_status', 'extracted_url_country',
        'extracted_url_coordinates', 'wayback_url', 'best_common',
        'best_common_prob', 'best_full', 'best_full_prob', 'best_name',
        'best_name_prob', 'confidence'
    ]

    df = pd.DataFrame(
        [[
            '123', 'The text', 'mmCIF, PDB', '0.987, 0.775',
            'Protein Data Bank', '0.717', 'http://www.pdb.org/', '200', 'US',
            '(34.22,-118.24)', '', 'mmCIF', '0.987', 'Protein Data Bank',
            '0.717', 'mmCIF', '0.987', ''
        ],
         [
             '456', 'More text.', '', '', 'SBASE', '0.648',
             'http://www.icgeb.trieste.it/sbase', '301', '', '', '', '', '',
             'SBASE', '0.648', 'SBASE', '0.648', 'manual_review'
         ],
         [
             '147', 'Wow.', 'TwoURLS', '0.998', '', '',
             'http://website.com, http://db.org', '200, 302', 'JP, GB',
             '(35.67,139.65), (52.20,0.13)', '', 'TwoURLS', '0.998', '', '',
             'TwoURLS', '0.998', ''
         ],
         [
             '369', 'The cat drank wine', 'PDB', '0.963', 'Protein Data Bank',
             '0.964', 'http://www.pdb.org/', '200', 'US', '(34.22,-118.24)',
             '', 'PDB', '0.963', 'Protein Data Bank', '0.964',
             'Protein Data Bank', '0.964', 'manual_review'
         ],
         [
             '741', 'Almost 7eleven', 'PDB', '0.983', 'Protein Data Bank',
             '0.964', 'http://www.pdb.org/', '200', 'US', '(34.22,-118.24)',
             '', 'PDB', '0.983', 'Protein Data Bank', '0.964', 'PDB', '0.983',
             ''
         ],
         [
             '852', 'Chihiro', 'PDB', '0.963', 'Protein Data Bank', '0.984',
             'http://www.pdb.org/', '200', 'US', '(34.22,-118.24)', '', 'PDB',
             '0.963', 'Protein Data Bank', '0.984', 'Protein Data Bank',
             '0.984', ''
         ]],
        columns=columns)

    return df


# ---------------------------------------------------------------------------
def deduplicate(df: pd.DataFrame, thresh: float, common: bool, full: bool,
                url: bool) -> pd.DataFrame:
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

    df = df.drop(
        ['text', 'common_name', 'common_prob', 'full_name', 'full_prob'],
        axis='columns')
    df['best_common_prob'] = df['best_common_prob'].astype(str)
    df['best_full_prob'] = df['best_full_prob'].astype(str)
    df['best_name_prob'] = df['best_name_prob'].astype(str)

    # Do not deduplicate/aggregate rows flagged for review
    low_df = df[df['best_name_prob'].astype(float) < thresh]
    high_df = df[df['best_name_prob'].astype(float) >= thresh]

    if not any([common, full, url]):
        duplicate_name = high_df.duplicated('best_name', keep=False)

        unique_name_df = high_df[~duplicate_name]
        same_name_df = high_df[duplicate_name]
        same_name_df['sum_citations'] = 0

        same_name_df = (same_name_df.groupby(['best_name']).agg({
            'ID':
            join_commas,
            'extracted_url':
            join_commas,
            'extracted_url_status':
            join_commas,
            'extracted_url_country':
            join_commas,
            'extracted_url_coordinates':
            join_commas,
            'best_common':
            join_commas,
            'best_common_prob':
            join_commas,
            'best_full':
            join_commas,
            'best_full_prob':
            join_commas,
            'confidence':
            join_commas,
            'sum_citations':
            len
        }).reset_index())

        same_name_df = wrangle_names(same_name_df, 'best_common',
                                     'best_common_prob', 'best_full',
                                     'best_full_prob')

        unique_name_df['sum_citations'] = 1
        low_df['sum_citations'] = 1

        out_df = pd.concat([low_df, unique_name_df, same_name_df])

        return out_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
def test_deduplicate(raw_data: pd.DataFrame) -> None:
    """ Test deduplicate() """

    out_ids = pd.Series(['456', '123', '147', '741', '369, 852'], name='ID')
    out_citations = pd.Series([1, 1, 1, 1, 2], name='sum_citations')

    return_df = deduplicate(raw_data, 0.9, False, False, False)

    assert_series_equal(return_df['ID'], out_ids)
    assert_series_equal(return_df['sum_citations'], out_citations)


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
