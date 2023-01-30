#!/usr/bin/env python3
"""
Purpose: Extract country information from affiliations and
         make IP countries consistent
Authors: Kenneth Schackart
"""

import argparse
import os
import re
from typing import NamedTuple, TextIO

import pandas as pd
import pycountry
from pandas.testing import assert_series_equal

from inventory_utils.custom_classes import CustomHelpFormatter
from inventory_utils.wrangling import join_commas

pd.options.mode.chained_assignment = None


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    file: TextIO
    out_dir: str
    country_format: str


# ---------------------------------------------------------------------------
def get_args() -> Args:
    """ Parse command-line arguments """

    desc = ('Extract country information from affiliations '
            'and make IP countries consistent')
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=CustomHelpFormatter)

    parser.add_argument('file',
                        metavar='FILE',
                        type=argparse.FileType('rt', encoding='ISO-8859-1'),
                        help='CSV file of inventory')
    parser.add_argument('-o',
                        '--out-dir',
                        metavar='DIR',
                        type=str,
                        default='out/',
                        help='Output directory')
    parser.add_argument('-f',
                        '--format',
                        metavar='FMT',
                        type=str,
                        default='alpha-3',
                        choices=['alpha-2', 'alpha-3', 'full', 'numeric'],
                        help='ISO 3166-1 Country Code output format')

    args = parser.parse_args()

    return Args(args.file, args.out_dir, args.format)


# ---------------------------------------------------------------------------
def extract_countries(strings: pd.Series, country_format: str) -> pd.Series:
    """
    Extract country names from column of strings

    Parameters:
    `strings`: Column of strings that may contain country mentions
    `country_format`: Country code output format

    Return: Column of extracted country names
    """

    countries = []
    for string in strings:
        found_countries = []
        for country in pycountry.countries:
            for country_code in [
                    country.name,
                    country.alpha_3  # country.alpha_2
            ]:
                matches = re.findall(fr'\b{country_code}\b', string)

                if not matches:
                    continue

                if country_format == 'alpha-2':
                    found_country = country.alpha_2
                elif country_format == 'alpha-3':
                    found_country = country.alpha_3
                elif country_format == 'numeric':
                    found_country = country.numeric
                else:
                    found_country = country.name

                for _ in matches:
                    found_countries.append(found_country)

        countries.append(join_commas(found_countries))

    return pd.Series(countries)


# ---------------------------------------------------------------------------
def test_extract_countries() -> None:
    """ Test extract_countries() """

    in_col = pd.Series([
        'USA.', 'United States', 'US', 'The United States of America',
        '605014, India.', 'France', 'GB'
    ])

    # Can retrieve 2 character country codes
    out_col = pd.Series(['US', 'US', 'US', 'US', 'IN', 'FR', 'GB'])
    assert_series_equal(extract_countries(in_col, 'alpha-2'), out_col)

    # Can retrieve 3 character countrty codes
    out_col = pd.Series(['USA', 'USA', 'USA', 'USA', 'IND', 'FRA', 'GBR'])
    assert_series_equal(extract_countries(in_col, 'alpha-3'), out_col)

    # Can retrieve countrty names
    out_col = pd.Series([
        'United States', 'United States', 'United States', 'United States',
        'India', 'France', 'United Kingdom'
    ])
    assert_series_equal(extract_countries(in_col, 'full'), out_col)

    # Can retrieve numeric country codes
    out_col = pd.Series(['840', '840', '840', '840', '356', '250', '826'])
    assert_series_equal(extract_countries(in_col, 'numeric'), out_col)

    # Can retrieve multiple instances from single row
    # Returns empty string if none found
    in_col = pd.Series(['Slovenia and Singapore', 'Portugal', ''])
    out_col = pd.Series(['SGP, SVN', 'PRT', ''])
    assert_series_equal(extract_countries(in_col, 'alpha-3'), out_col)

    # Keeps the original number of instances
    in_col = pd.Series(['France, India, and France'])
    out_col = pd.Series(['FRA, FRA, IND'])
    assert_series_equal(extract_countries(in_col, 'alpha-3'), out_col)


# ---------------------------------------------------------------------------
def process_data(df: pd.DataFrame, country_format: str) -> pd.DataFrame:
    """
    Process manually reviewed data.

    Parameters:
    `df`: Manually reviewed dataframe
    `country_format`: Country code output format

    Return: Processed dataframe
    """

    df['affiliation_countries'] = extract_countries(df['affiliation'],
                                                    country_format)
    df['extracted_url_country'] = extract_countries(
        df['extracted_url_country'], country_format)

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

    in_df = pd.read_csv(args.file, dtype=str).fillna('')

    out_df = process_data(in_df, args.country_format)

    out_df.to_csv(outfile, index=False)

    print(f'Done. Wrote output to {outfile}.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
