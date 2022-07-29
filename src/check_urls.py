#!/usr/bin/env python3
"""
Purpose: Check URLs
Authors: Kenneth Schackart
"""

import argparse
from functools import partial
import multiprocessing as mp
import os
import time
from typing import NamedTuple, Optional, TextIO, Union, cast
from multiprocessing.pool import Pool

import pandas as pd
import requests
from pandas.testing import assert_frame_equal

from utils import CustomHelpFormatter


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    file: TextIO
    out_dir: str
    num_tries: int
    wait: int
    ncpu: Optional[int]


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
        description=('Check extracted URL statuses'),
        formatter_class=CustomHelpFormatter)

    parser.add_argument('file',
                        metavar='FILE',
                        type=argparse.FileType('rt', encoding='ISO-8859-1'),
                        help='CSV File with extracted_url column')
    parser.add_argument('-o',
                        '--out-dir',
                        metavar='DIR',
                        type=str,
                        default='out/',
                        help='Output directory')
    parser.add_argument('-n',
                        '--num-tries',
                        metavar='INT',
                        type=int,
                        default=3,
                        help='Number of tried for checking URL')
    parser.add_argument('-w',
                        '--wait',
                        metavar='TIME',
                        type=int,
                        default=500,
                        help='Time (ms) to wait between tries')
    parser.add_argument('-t',
                        '--ncpu',
                        metavar='CPU',
                        type=int,
                        help=('Number of CPUs for parallel '
                              'processing (default: all)'))

    args = parser.parse_args()

    return Args(args.file, args.out_dir, args.num_tries, args.wait, args.ncpu)


# ---------------------------------------------------------------------------
def expand_url_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the URL column, by creating a row per URL.
    
    `df`: Dataframe with extracted_url column
    
    Return: Dataframe with row per URL
    """

    df['extracted_url'] = df['extracted_url'].str.split(', ')

    df = df.explode('extracted_url')

    df.reset_index(drop=True, inplace=True)

    return df


# ---------------------------------------------------------------------------
def test_expand_url_col() -> None:
    """ Test expand_url_col() """

    in_df = pd.DataFrame(
        [[123, 'Some text', 'https://www.google.com, http://google.com'],
         [789, 'Foo', 'https://www.amazon.com/afbadfbnvbadfbaefbnaegn']],
        columns=['ID', 'text', 'extracted_url'])

    out_df = pd.DataFrame(
        [[123, 'Some text', 'https://www.google.com'],
         [123, 'Some text', 'http://google.com'],
         [789, 'Foo', 'https://www.amazon.com/afbadfbnvbadfbaefbnaegn']],
        columns=['ID', 'text', 'extracted_url'])

    assert_frame_equal(expand_url_col(in_df), out_df)


# ---------------------------------------------------------------------------
def get_pool(ncpu: Optional[int]) -> Pool:
    """
    Get Pool for multiprocessing.
    
    Parameters:
    `ncpu`: Number of CPUs to use, if not specified detect number available
    
    Return:
    `Pool` using `ncpu` or number of available CPUs
    """

    n_cpus = ncpu if ncpu else mp.cpu_count()

    print(f'Running with {n_cpus} processes')

    return mp.Pool(n_cpus)


# ---------------------------------------------------------------------------
def make_filename(out_dir: str, infile_name: str) -> str:
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

    assert make_filename(
        'out/checked_urls',
        'out/urls/predictions.csv') == ('out/checked_urls/predictions.csv')


# ---------------------------------------------------------------------------
def request_url(url: str) -> Union[int, str]:
    """
    Check a URL once using try-except to catch exceptions
    
    Parameters:
    `url`: URL string
    
    Return: Status code or error message
    """

    try:
        r = requests.head(url)
    except requests.exceptions.RequestException as err:
        return str(err)

    return r.status_code


# ---------------------------------------------------------------------------
def test_request_url() -> None:
    """ Test request_url() """

    # Hopefully, Google doesn't disappear, if it does use a different URL
    assert request_url('https://www.google.com') == 200

    # Bad URLs
    assert request_url('http://google.com') == 301
    assert request_url('https://www.amazon.com/afbadfbnvbadfbaefbnaegn') == 404

    # Runtime exception
    assert request_url('adflkbndijfbn') == (
        "Invalid URL 'adflkbndijfbn': No scheme supplied. "
        "Perhaps you meant http://adflkbndijfbn?")


# ---------------------------------------------------------------------------
def check_url(url: str, num_tries: int, wait: int) -> Union[int, str]:
    """
    Try requesting URL the specified number of tries, returning 200
    if it succeeds at least once

    Parameters:
    `url`: URL string
    `num_tries`: Number of times to try requesting URL
    `wait`: Wait time between tries in ms

    Return: Status code or error message
    """

    for _ in range(num_tries):
        status = request_url(url)
        if status == 200:  # Status code was returned
            break
        time.sleep(wait / 1000)

    return status


# ---------------------------------------------------------------------------
def test_check_url() -> None:
    """ Test check_url() """

    assert check_url('https://www.google.com', 3, 0) == 200

    # Bad URLs
    assert check_url('http://google.com', 3, 0) == 301
    assert check_url('https://www.amazon.com/afbadfbnvbadfbaefbnaegn', 3,
                     250) == 404

    # Runtime exception
    assert check_url(
        'adflkbndijfbn', 3,
        250) == ("Invalid URL 'adflkbndijfbn': No scheme supplied. "
                 "Perhaps you meant http://adflkbndijfbn?")


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
    retrieved = check_wayback('example.com')

    assert retrieved.status is not None
    assert retrieved.url is not None
    assert retrieved.timestamp is not None

    # Valid URL, but not present as a snapshot

    # Invalid URL
    assert check_wayback('aegkbnwefnb') == WayBackSnapshot(None, None, None)


# ---------------------------------------------------------------------------
def check_urls(df: pd.DataFrame, ncpu: Optional[int], num_tries: int,
               wait: int) -> pd.DataFrame:
    """
    Check all URLs in df
    
    Parameters:
    `df`: Dataframe with url column
    
    Return: Dataframe
    """

    check_url_part = partial(check_url, num_tries=num_tries, wait=wait)

    with get_pool(ncpu) as pool:
        df['extracted_url_status'] = pool.map_async(check_url_part,
                                                    df['extracted_url']).get()
        df['wayback_url'] = pool.map_async(check_wayback,
                                           df['extracted_url']).get().append

    return df


# ---------------------------------------------------------------------------
def test_check_urls() -> None:
    """ Test check_urls() """

    in_df = pd.DataFrame(
        [[123, 'Some text', 'https://www.google.com'],
         [456, 'More text', 'http://google.com'],
         [789, 'Foo', 'https://www.amazon.com/afbadfbnvbadfbaefbnaegn']],
        columns=['ID', 'text', 'extracted_url'])

    out_df = in_df = pd.DataFrame(
        [[123, 'Some text', 'https://www.google.com', 200],
         [456, 'More text', 'http://google.com', 301],
         [789, 'Foo', 'https://www.amazon.com/afbadfbnvbadfbaefbnaegn', 404]],
        columns=['ID', 'text', 'extracted_url', 'extracted_url_status'])

    returned_df = check_urls(in_df, None, 3, 0)
    returned_df.sort_values('ID', inplace=True)

    assert_frame_equal(returned_df, out_df)


# ---------------------------------------------------------------------------
def regroup_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regroup dataframe to contain one row per article, columns may contain
    list elements
    
    `df`: Dataframe with one row per URL
    
    Return: Dataframe with one row per article
    """

    df['extracted_url_status'] = df['extracted_url_status'].astype(str)

    out_df = (df.groupby(['ID', 'text']).agg({
        'extracted_url':
        lambda x: ', '.join(x),
        'extracted_url_status':
        lambda x: ', '.join(x)
    }).reset_index())

    return out_df


# ---------------------------------------------------------------------------
def test_regroup_df() -> None:
    """ Test regroup_df() """

    in_df = pd.DataFrame(
        [[123, 'Some text', 'https://www.google.com', 200],
         [123, 'Some text', 'http://google.com', 301],
         [789, 'Foo', 'https://www.amazon.com/afbadfbnvbadfbaefbnaegn', 404]],
        columns=['ID', 'text', 'extracted_url', 'extracted_url_status'])

    out_df = pd.DataFrame([[
        123, 'Some text', 'https://www.google.com, http://google.com',
        '200, 301'
    ], [789, 'Foo', 'https://www.amazon.com/afbadfbnvbadfbaefbnaegn', '404']],
                          columns=[
                              'ID', 'text', 'extracted_url',
                              'extracted_url_status'
                          ])

    assert_frame_equal(regroup_df(in_df), out_df)


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()
    out_dir = args.out_dir

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    df = pd.read_csv(args.file)

    df = expand_url_col(df)
    df = check_urls(df, args.ncpu, args.num_tries, args.wait)
    df = regroup_df(df)

    df.to_csv(make_filename(out_dir, args.file), index=False)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
