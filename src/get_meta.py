#!/usr/bin/env python3
"""
Purpose: Get metadata from EuropePMC query
Authors: Kenneth Schackart
"""

import argparse
import os
import re
from collections import defaultdict
from typing import NamedTuple, TextIO, Tuple, cast

import pandas as pd
import pycountry
import requests
from pandas.testing import assert_series_equal

from utils import CustomHelpFormatter


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    file: TextIO
    out_dir: str


# ---------------------------------------------------------------------------
def get_args() -> Args:
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description=('Get metadata from EuropePMC query'),
        formatter_class=CustomHelpFormatter)

    parser.add_argument('file',
                        metavar='FILE',
                        type=argparse.FileType('rt', encoding='ISO-8859-1'),
                        help='CSV File with ID column for articles')
    parser.add_argument('-o',
                        '--out-dir',
                        metavar='DIR',
                        type=str,
                        default='out/',
                        help='Output directory')

    args = parser.parse_args()

    return Args(args.file, args.out_dir)


# ---------------------------------------------------------------------------
def make_filenames(outdir: str) -> Tuple[str, str]:
    '''
    Make filenames for output csv file and last date text file

    Parameters:
    `outdir`: Output directory

    Return: Tuple of csv and txt filenames
    '''

    csv_out = os.path.join(outdir, 'query_results.csv')
    txt_out = os.path.join(outdir, 'last_query_date.txt')

    return csv_out, txt_out


# ---------------------------------------------------------------------------
def test_make_filenames() -> None:
    """ Test make_filenames() """

    assert make_filenames('data/new_query') == (
        'data/new_query/query_results.csv',
        'data/new_query/last_query_date.txt')


# ---------------------------------------------------------------------------
def clean_results(results: dict) -> pd.DataFrame:
    """
    Retrieve the metadata from results of query

    Parameters:
    `results`: JSON-encoded response (nested dictionary)

    Return: Dataframe of results
    """

    parsed_info = defaultdict(list)
    for paper in results.get('resultList').get('result'):  # type: ignore
        parsed_info['ids'].append(paper.get('id'))
        parsed_info['titles'].append(paper.get('title'))
        parsed_info['abstracts'].append(paper.get('abstractText'))
        parsed_info['affiliations'].append(paper.get('affiliation'))

        authors = []
        for author in paper.get('authorList', {}).get('author', {}):
            if author:
                authors.append(author.get('fullName', ''))
            else:
                authors.append('')
        parsed_info['authors'].append(', '.join(authors))

        grant_ids = []
        agencies = []
        for grant in paper.get('grantsList', {}).get('grant', {}):
            if grant:
                grant_ids.append(grant.get('grantID', ''))
                agencies.append(grant.get('agency', ''))
            else:
                grant_ids.append('')
                agencies.append('')
        parsed_info['grant_ids'].append(', '.join(
            [grant_id for grant_id in grant_ids if grant_id]))
        parsed_info['agencies'].append(', '.join(
            [agency for agency in agencies if agency]))

    return pd.DataFrame({
        'ID': parsed_info['ids'],
        'affiliation': parsed_info['affiliations'],
        'authors': parsed_info['authors'],
        'grant_ids': parsed_info['grant_ids'],
        'grant_agencies': parsed_info['agencies']
    })


# ---------------------------------------------------------------------------
def run_query(ids: pd.Series) -> pd.DataFrame:
    """
    Run query on EuropePMC API

    Parameters:
    `ids`: Dataframe ID column

    Return: `DataFrame` of returned article information
    """

    query = ' OR '.join(set(ids))
    prefix = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query='
    suffix = '&resultType=core&fromSearchPost=false&format=json'
    url = prefix + query + suffix

    # Not using try-except because if there is an exception it is not
    # because there is not an archived version, it means the API
    # has changed.
    results = requests.get(url)
    if results.status_code != requests.codes.ok:  # pylint: disable=no-member
        results.raise_for_status()

    results_json = cast(dict, results.json())

    return clean_results(results_json)


# ---------------------------------------------------------------------------
def extract_countries(affiliations: pd.Series) -> pd.Series:
    """
    Extract country names from affiliations column
    
    Parameters:
    `affiliations`: Column of affiliations
    
    Return: column of extracted country names
    """

    countries = []
    for affiliation in affiliations:
        found_countries = []
        for country in pycountry.countries:
            if any([
                    re.search(fr'\b{x}\b', affiliation)
                    for x in [country.name, country.alpha_3, country.alpha_2]
            ]):
                found_countries.append(country.name)
        countries.append(', '.join(found_countries))

    return pd.Series(countries)


# ---------------------------------------------------------------------------
def test_extract_countries() -> None:
    """ Test extract_countries() """

    in_col = pd.Series([
        'USA.', 'United States', 'US', 'The United States of America',
        '605014, India.', 'France', 'UK'
    ])

    out_col = pd.Series([
        'United States', 'United States', 'United States', 'United States',
        'India', 'France', 'UK'
    ])

    assert_series_equal(extract_countries(in_col), out_col)


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()
    out_dir = args.out_dir

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    df = pd.read_csv(args.file)
    df['ID'] = df['ID'].astype(str)

    results = run_query(df['ID'])
    results['ID'] = results['ID'].astype(str)

    all_info = pd.merge(df, results, how='inner', on='ID')

    all_info['countries'] = extract_countries(all_info['affiliation'])

    all_info.to_csv(os.path.join(out_dir, 'meta.csv'), index=False)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
