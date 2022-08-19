#!/usr/bin/env python3
"""
Purpose: Run query on EuropePMC
Authors: Ana Maria Istrate and Kenneth Schackart
"""

import argparse
import os
import re
from datetime import datetime
from typing import NamedTuple, Tuple, cast

import pandas as pd
import requests

from inventory_utils.custom_classes import CustomHelpFormatter


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    query: str
    from_date: str
    to_date: str
    out_dir: str


# ---------------------------------------------------------------------------
def get_args() -> Args:
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description=('Query EuropePMC to retrieve articles. '
                     'Saves csv of results and file of today\'s date'),
        formatter_class=CustomHelpFormatter)

    parser.add_argument('query',
                        metavar='QUERY',
                        type=str,
                        help='EuropePMC query to run (file or string)')
    parser.add_argument('-f',
                        '--from-date',
                        metavar='DATE',
                        type=str,
                        default='2011',
                        help='Articles published after (file or string)')
    parser.add_argument('-t',
                        '--to-date',
                        metavar='DATE',
                        type=str,
                        default=None,
                        help='Articles published before (default: today)')
    parser.add_argument('-o',
                        '--out-dir',
                        metavar='DIR',
                        type=str,
                        default='out/',
                        help='Output directory')

    args = parser.parse_args()

    if os.path.isfile(args.query):
        args.query = open(args.query).read()
    if os.path.isfile(args.from_date):
        args.from_date = open(args.from_date).read()

    date_pattern = re.compile(
        r'''^           # Beginning of date string
            [\d]{4}     # Must start wwith 4 digit year
            (-[\d]{2}   # Optionally 2 digit month
            (-[\d]{2})? # Optionally 2 digit day
            )?          # Finish making month optional
            $           # Followed by nothing else
            ''', re.X)
    for date in [args.from_date, args.to_date]:
        if not re.match(date_pattern, args.date):
            parser.error(f'Last date "{date}" must be one of:\n'
                         '\t\t\tYYYY\n'
                         '\t\t\tYYYY-MM\n'
                         '\t\t\tYYYY-MM-DD')

    return Args(args.query, args.from_date, args.to_date, args.out_dir)


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
    Retrieve the PMIDs, titles, and abstracts from results of query

    Parameters:
    `results`: JSON-encoded response (nested dictionary)

    Return: Dataframe of results
    """

    pmids = []
    titles = []
    abstracts = []
    for paper in results.get('resultList').get('result'):  # type: ignore
        pmids.append(paper.get('pmid'))
        titles.append(paper.get('title'))
        abstracts.append(paper.get('abstractText'))

    return pd.DataFrame({'id': pmids, 'title': titles, 'abstract': abstracts})


# ---------------------------------------------------------------------------
def run_query(query: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Run query on EuropePMC API

    Parameters:
    `query`: Query to use
    `from_date`: Articles published after this date
    `to_date`: Articles published after this date

    Return: `DataFrame` of returned titles and abstracts
    """

    query = query.format(from_date, to_date)

    prefix = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query='
    suffix = '&resultType=core&fromSearchPost=false&format=json'
    url = prefix + query + suffix

    results = requests.get(url)
    if results.status_code != requests.codes.ok:  # pylint: disable=no-member
        results.raise_for_status()

    results_json = cast(dict, results.json())

    return clean_results(results_json)


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()
    out_dir = args.out_dir

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_df, date_out = make_filenames(out_dir)

    if not args.to_date:
        to_date = datetime.today().strftime(r'%Y-%m-%d')
    else:
        to_date = args.to_date

    results = run_query(args.query, args.from_date, to_date)

    results.to_csv(out_df, index=False)
    print(to_date, file=open(date_out, 'wt'))

    print(f'Done. Wrote 2 files to {out_dir}.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
