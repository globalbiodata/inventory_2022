#!/usr/bin/env python3
"""
Purpose: Submit URLs to WayBack Machine if they are missing
Authors: AKenneth Schackart
"""

import argparse
import os
from subprocess import getstatusoutput
from typing import List, NamedTuple, TextIO

import pandas as pd
from pandas.testing import assert_frame_equal

from inventory_utils.custom_classes import CustomHelpFormatter


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    file: TextIO
    key: str


# ---------------------------------------------------------------------------
def get_args() -> Args:
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description=('Submit request to capture any URLs'
                     ' not present in WayBack Machine'),
        formatter_class=CustomHelpFormatter)

    parser.add_argument('file',
                        metavar='FILE',
                        type=argparse.FileType('rt', encoding='ISO-8859-1'),
                        help='Inventory file')
    parser.add_argument('-k',
                        '--key',
                        metavar='KEY|FILE',
                        type=str,
                        required=True,
                        help='Internet Archive user secret key')

    args = parser.parse_args()

    if os.path.isfile(args.key):
        args.key = open(args.key).read()

    return Args(args.file, args.key)


# ---------------------------------------------------------------------------
def expand_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the URL and wayback columns, by creating a row per URL.

    `df`: Dataframe with extracted_url and wayback_url columns

    Return: Dataframe with row per URL
    """

    df['extracted_url'] = df['extracted_url'].str.split(', ')
    df['wayback_url'] = df['wayback_url'].str.split(', ')

    df = df.explode(['extracted_url', 'wayback_url'])

    df.reset_index(drop=True, inplace=True)

    return df


# ---------------------------------------------------------------------------
def test_expand_cols() -> None:
    """ Test expand_cols() """

    in_df = pd.DataFrame(
        [['url_1, url_2', 'wb_1, no_wayback'], ['url_3', 'wb_3']],
        columns=['extracted_url', 'wayback_url'])

    out_df = pd.DataFrame(
        [['url_1', 'wb_1'], ['url_2', 'no_wayback'], ['url_3', 'wb_3']],
        columns=['extracted_url', 'wayback_url'])

    assert_frame_equal(expand_cols(in_df), out_df)


# ---------------------------------------------------------------------------
def get_missing_urls(df: pd.DataFrame) -> List[str]:
    """
    Find URLs that are not in the WayBack machine

    Parameters:
    `df`: Dataframe with `wayback_url` column

    Return: List of urls not present in WayBack machine
    """
    df = expand_cols(df)
    df = df[df['wayback_url'] == 'no_wayback']

    return list(df['extracted_url'])


# ---------------------------------------------------------------------------
def test_get_missing_urls() -> None:
    """ Test get_missing_urls """

    # Returns missing URLs
    in_df = pd.DataFrame([['url_1', 'wb_1'], ['url_2', 'no_wayback']],
                         columns=['extracted_url', 'wayback_url'])
    missing_urls = ['url_2']
    assert get_missing_urls(in_df) == missing_urls

    # Is okay with extra columns
    in_df = pd.DataFrame(
        [['123', 'url_1', 'wb_1'], ['456', 'url_2', 'no_wayback']],
        columns=['ID', 'extracted_url', 'wayback_url'])
    missing_urls = ['url_2']
    assert get_missing_urls(in_df) == missing_urls

    # Can return multiple URLs per resource
    in_df = pd.DataFrame(
        [['url_1', 'wb_1'], ['url_2, url_3', 'no_wayback, no_wayback'],
         ['url_4, url_5', 'wb_4, no_wayback']],
        columns=['extracted_url', 'wayback_url'])
    missing_urls = ['url_2', 'url_3', 'url_5']
    assert get_missing_urls(in_df) == missing_urls


# ---------------------------------------------------------------------------
def get_command(url: str, key: str) -> str:
    """
    Get submission command for a URL

    Parameters:
    `url`: URL to submit
    `key`: Internet Archive user secret key

    Return: shell command for submitting
    """

    command = ('curl -X POST '
               '-H "Accept: application/json" '
               f'-H "Authorization: LOW myaccesskey:{key}" '
               '-d\'url={}\' https://web.archine.org/save')

    command = command.format(url)

    return command


# ---------------------------------------------------------------------------
def test_get_command() -> None:
    """ Test get_command() """

    key = 'foo'
    url = 'bar'

    expected = ('curl -X POST '
                '-H "Accept: application/json" '
                '-H "Authorization: LOW myaccesskey:foo" '
                '-d\'url=bar\' https://web.archine.org/save')

    assert get_command(url, key) == expected


# ---------------------------------------------------------------------------
def submit_urls(urls: List[str], key: str) -> None:
    """
    Submit URLs for capture by WayBack Machine

    Parameters:
    `urls`: List of URLs to submit
    `key`: Internet Archive user secret key
    """

    print(f'Submitting {len(urls)} urls.')

    for url in urls:
        print(f'Submitting {url}... ', end='')
        command = get_command(url, key)
        retval, out = getstatusoutput(command)
        if retval == 0:
            print('Done.')
        else:
            print('Non-zero return value, see output:')
            print(out)


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()

    in_df = pd.read_csv(args.file, dtype=str)

    missing_urls = get_missing_urls(in_df)

    submit_urls(missing_urls, args.key)

    print('Done.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
