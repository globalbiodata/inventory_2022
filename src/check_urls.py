#!/usr/bin/env python3
"""
Purpose: Check URLs
Authors: Kenneth Schackart
"""

import argparse
import os
import re
from collections import defaultdict
from typing import NamedTuple, Optional, TextIO, Tuple, cast

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
class WayBackSnapshot(NamedTuple):
    """
    Information about a WayBack Archive Snapshot
    
    `url`: URL of Snapshot on WayBack Machine
    `timestamp`: Timestamp of archive
    `status`: Snapshot status
    """
    url: Optional[str]
    timestamp: Optional[str]
    status: Optional[str]


# ---------------------------------------------------------------------------
def get_args() -> Args:
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description=('Get metadata from EuropePMC query'),
        formatter_class=CustomHelpFormatter)

    parser.add_argument('file',
                        metavar='FILE',
                        type=argparse.FileType('rt', encoding='ISO-8859-1'),
                        help='CSV File with name and full_name columns')
    parser.add_argument('-o',
                        '--out-dir',
                        metavar='DIR',
                        type=str,
                        default='out/',
                        help='Output directory')

    args = parser.parse_args()

    return Args(args.file, args.out_dir)


# ---------------------------------------------------------------------------
def make_filenames(out_dir: str, infile_name: str) -> str:
    '''
    Make filename for output reusing input file's basename

    Parameters:
    `outdir`: Output directory

    Return: Output filename
    '''

    return os.path.join(out_dir, os.path.basename(infile_name))


# ---------------------------------------------------------------------------
def test_make_filenames() -> None:
    """ Test make_filenames() """

    assert make_filenames(
        'out/checked_urls',
        'out/urls/predictions.csv') == ('out/checked_urls/predictions.csv')


# ---------------------------------------------------------------------------
def url_ok(url: str) -> bool:
    """
    Check that a URL returns status code 200
    
    Parameters:
    `url`: URL string
    
    Return: True if URL is good, False otherwise
    """

    try:
        r = requests.head(url)
    except:
        return False

    return r.status_code == 200


# ---------------------------------------------------------------------------
def test_url_ok() -> None:
    """ Test url_ok() """

    # Hopefully, Google doesn't disappear, if it does use a different URL
    assert url_ok('https://www.google.com')

    # Bad URLs
    # 301
    assert not url_ok('http://google.com')
    # 404
    assert not url_ok('https://www.amazon.com/afbadfbnvbadfbaefbnaegn')
    # Runtime exception
    assert not url_ok('adflkbndijfbn')


# ---------------------------------------------------------------------------
def check_wayback(url: str) -> WayBackSnapshot:
    """
    Check the WayBack Machine for an archived version of requested URL.
    
    Parameters:
    `url`: URL to check
    
    Return: A `WayBackSnapshot` NamedTuple
    with attributes `url`, `timestamp`, and `status`
    """

    # Not using try-except because if there is an exception it is not
    # because there is not an archived version, it means the API
    # has changed.
    r = requests.get(f'http://archive.org/wayback/available?url={url}')

    returned_dict = cast(dict, r.json())
    snapshots = cast(dict, returned_dict.get('archived_snapshots'))

    if not snapshots:
        return WayBackSnapshot(None, None, None)

    snapshot = cast(dict, snapshots.get('closest'))

    return WayBackSnapshot(
        *[snapshot.get(key) for key in ['url', 'timestamp', 'status']])


# ---------------------------------------------------------------------------
def test_check_wayback() -> None:
    """ Test check_wayback() """

    # Example from their website
    expected = WayBackSnapshot(
        'http://web.archive.org/web/20220719133038/https://example.com/',
        '20220719133038', '200')

    assert check_wayback('example.com') == expected

    # Valid URL, but not present as a snapshot

    # Invalid URL
    assert check_wayback('aegkbnwefnb') == WayBackSnapshot(None, None, None)


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()
    out_dir = args.out_dir

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    df = pd.read_csv(args.file)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
