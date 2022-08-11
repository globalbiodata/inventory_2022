#!/usr/bin/env python3
"""
Purpose: Check URLs
Authors: Kenneth Schackart
"""

import argparse
import logging
import multiprocessing as mp
import os
import re
import socket
import time
from functools import partial
from multiprocessing.pool import Pool
from typing import List, NamedTuple, Optional, TextIO, Union, cast

import pandas as pd
import requests
from pandas.testing import assert_frame_equal

from utils import CustomHelpFormatter

API_REQ_DICT = {
    'ipinfo': 'https://ipinfo.io/{}/json',
    'ip-api': 'http://ip-api.com/json/{}'
}


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    file: TextIO
    out_dir: str
    num_tries: int
    wait: int
    cores: Optional[int]
    verbose: bool


# ---------------------------------------------------------------------------
class URLStatus(NamedTuple):
    """
    URL and its returned status and location
    """
    url: str
    status: Union[str, int]
    country: str
    latitude: str
    longitude: str


# ---------------------------------------------------------------------------
class IPLocation(NamedTuple):
    """ IP address location """
    country: str
    latitude: str
    longitude: str


# ---------------------------------------------------------------------------
def get_args() -> Args:
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description=('Check extracted URL statuses and '
                     'see if snapsot is in WayBack Machine'),
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
                        help='Number of tries for checking URL')
    parser.add_argument('-w',
                        '--wait',
                        metavar='MS',
                        type=int,
                        default=500,
                        help='Time (ms) to wait between tries')
    parser.add_argument('-c',
                        '--cores',
                        metavar='CORE',
                        type=int,
                        help=('Number of cores for parallel '
                              'processing (default: all)'))
    parser.add_argument('-v',
                        '--verbose',
                        action='store_true',
                        help=('Run with debugging messages'))

    args = parser.parse_args()

    return Args(args.file, args.out_dir, args.num_tries, args.wait, args.cores,
                args.verbose)


# ---------------------------------------------------------------------------
def expand_url_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the URL column, by creating a row per URL.

    `df`: Dataframe with extracted_url column

    Return: Dataframe with row per URL
    """
    logging.debug('Expanding URL column. One row per URL')

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
def get_pool(cores: Optional[int]) -> Pool:
    """
    Get Pool for multiprocessing.

    Parameters:
    `cores`: Number of CPUs to use, if not specified detect number available

    Return:
    `Pool` using `cores` or number of available CPUs
    """

    n_cpus = cores if cores else mp.cpu_count()

    logging.debug('Running with %d processes', n_cpus)

    return mp.Pool(n_cpus)  # pylint: disable=consider-using-with


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
def extract_domain(url: str) -> str:
    """
    Extract domain name from URL

    Parameters:
    `url`: URL string

    Return: Domain string
    """

    domain = re.sub('https?://', '', url)
    domain = re.sub('/.*$', '', domain)

    return domain


# ---------------------------------------------------------------------------
def test_extract_domain() -> None:
    """ Test extract_domain() """

    assert extract_domain('https://www.google.com') == 'www.google.com'
    assert extract_domain('www.google.com') == 'www.google.com'
    assert extract_domain(
        'http://proteome.moffitt.org/QUAD/') == 'proteome.moffitt.org'


# ---------------------------------------------------------------------------
def query_ip(ip: str, api: str) -> IPLocation:
    """
    Query an API trying to find location from IP address

    Parameters:
    `ip`: IP address to check
    `api`: API to query

    Return: Location or empty string
    """

    logging.debug('Querying %s.', api)
    query_template = API_REQ_DICT[api]

    r = requests.get(query_template.format(ip), verify=True)

    if r.status_code != 200:
        return IPLocation('', '', '')

    data = cast(dict, r.json())

    country = data.get('country', '')
    latitude, longitude = '', ''

    if api == 'ipinfo':
        lat_lon = cast(str, data.get('loc', ''))
        lat_lon_split = lat_lon.split(',')
        if len(lat_lon_split) == 2:
            latitude, longitude = lat_lon_split
    elif api == 'ip-api':
        latitude = str(data.get('lat', ''))
        longitude = str(data.get('lon', ''))

    logging.debug('Obtained IP address location: %s', IPLocation.country)
    return IPLocation(country, latitude, longitude)


# ---------------------------------------------------------------------------
def get_location(url: str) -> IPLocation:
    """
    Get location (country) of URL by first fetching the IP address of
    the connection, then searching for location of that IP address

    Parameters:
    `url`: URL to search

    Return: Location or empty string
    """

    logging.debug('Attempting to determine IP address of %s', url)
    ip = socket.gethostbyname(extract_domain(url))

    if not ip:
        logging.debug('IP address for %s could not be determined', url)
        return IPLocation('', '', '')

    logging.debug('IP address found: %s.', ip)
    logging.debug('Attempting to geolocate IP address.')
    location = query_ip(ip, 'ipinfo')

    if '' in [location.country, location.latitude, location.longitude]:
        location = query_ip(ip, 'ip-api')

    logging.debug('Final location information for %s, %s', ip,
                  IPLocation.country)
    return location


# ---------------------------------------------------------------------------
def test_get_location() -> None:
    """ Test get_location() """

    location = get_location('https://google.com')
    assert location.country != ''

    location = get_location('google.com')
    assert location.country != ''


# ---------------------------------------------------------------------------
def check_url(url: str, num_tries: int, wait: int) -> URLStatus:
    """
    Try requesting URL the specified number of tries, returning 200
    if it succeeds at least once

    Parameters:
    `url`: URL string
    `num_tries`: Number of times to try requesting URL
    `wait`: Wait time between tries in ms

    Return: `URLStatus` (URL and its status code or error message)
    """

    location = IPLocation('', '', '')
    for i in range(num_tries):
        logging.debug('Requesting %s for the %d%s time', url, i + 1, {
            '1': 'st',
            '2': 'nd',
            '3': 'rd'
        }.get(str(i + 1)[-1], 'th'))
        status = request_url(url)
        logging.debug('Returned status for %s: %s', url, str(status))
        if status == 200:
            location = get_location(url)
            break
        time.sleep(wait / 1000)

    return URLStatus(url, status, location.country, location.latitude,
                     location.longitude)


# ---------------------------------------------------------------------------
def test_check_url() -> None:
    """ Test check_url() """

    url_status = check_url('https://www.google.com', 3, 0)
    assert url_status.url == 'https://www.google.com'
    assert url_status.status == 200

    # Bad URLs
    url_status = check_url('http://google.com', 3, 0)
    assert url_status.url == 'http://google.com'
    assert url_status.status == 301
    assert url_status.country == ''

    url_status = check_url('https://www.amazon.com/afbadffbaefbnaegn', 3, 0)
    assert url_status.url == 'https://www.amazon.com/afbadffbaefbnaegn'
    assert url_status.status == 404
    assert url_status.country == ''

    # Runtime exception
    url_status = check_url('adflkbndijfbn', 3, 250)
    assert url_status.url == 'adflkbndijfbn'
    assert url_status.status == (
        "Invalid URL 'adflkbndijfbn': No scheme supplied. "
        "Perhaps you meant http://adflkbndijfbn?")
    assert url_status.country == ''


# ---------------------------------------------------------------------------
def merge_url_statuses(df: pd.DataFrame,
                       url_statuses: List[URLStatus]) -> pd.DataFrame:
    """
    Create column of URL statuses

    Parameters:
    `df`: Dataframe containing extracted_url column
    `url_statuses`: List of `URLStatus` objects

    Return: Same dataframe, with additional extracted_url_status column
    """

    url_dict = {
        x.url: {
            'status': x.status,
            'country': x.country,
            'latitude': x.latitude,
            'longitude': x.longitude
        }
        for x in url_statuses
    }

    df['extracted_url_status'] = df['extracted_url'].map(
        lambda x: url_dict[x]['status'])
    df['extracted_url_country'] = df['extracted_url'].map(
        lambda x: url_dict[x]['country'])
    df['extracted_url_latitude'] = df['extracted_url'].map(
        lambda x: url_dict[x]['latitude'])
    df['extracted_url_longitude'] = df['extracted_url'].map(
        lambda x: url_dict[x]['longitude'])

    return df


# ---------------------------------------------------------------------------
def test_merge_url_statuses() -> None:
    """ Test merge_url_statuses() """

    in_df = pd.DataFrame([[123, 'Some text', 'https://www.google.com'],
                          [456, 'More text', 'http://google.com']],
                         columns=['ID', 'text', 'extracted_url'])

    statuses = [
        URLStatus('http://google.com', 301, '', '', ''),
        URLStatus('https://www.google.com', 200, 'United States', '34.0522',
                  '-118.2437')
    ]

    out_df = pd.DataFrame([[
        123, 'Some text', 'https://www.google.com', 200, 'United States',
        '34.0522', '-118.2437'
    ], [456, 'More text', 'http://google.com', 301, '', '', '']],
                          columns=[
                              'ID', 'text', 'extracted_url',
                              'extracted_url_status', 'extracted_url_country',
                              'extracted_url_latitude',
                              'extracted_url_longitude'
                          ])

    assert_frame_equal(merge_url_statuses(in_df, statuses), out_df)


# ---------------------------------------------------------------------------
def check_wayback(url: str) -> str:
    """
    Check the WayBack Machine for an archived version of requested URL

    Parameters:
    `url`: URL to check

    Return: WayBack snapshot URL or "no_wayback"
    """

    # Not using try-except because if there is an exception it is not
    # because there is not an archived version, it means the API
    # has changed. The code must be updated then.
    r = requests.get(f'http://archive.org/wayback/available?url={url}',
                     headers={'User-agent': 'biodata_resource_inventory'})

    returned_dict = cast(dict, r.json())
    snapshots = cast(dict, returned_dict.get('archived_snapshots'))

    if not snapshots:
        return 'no_wayback'

    snapshot = cast(dict, snapshots.get('closest'))

    return snapshot.get('url', 'no_wayback')


# ---------------------------------------------------------------------------
def test_check_wayback() -> None:
    """ Test check_wayback() """

    # Example from their website
    assert check_wayback('example.com') != ''

    # Valid URL, but not present as a snapshot

    # Invalid URL
    assert check_wayback('aegkbnwefnb') == 'no_wayback'


# ---------------------------------------------------------------------------
def check_urls(df: pd.DataFrame, cores: Optional[int], num_tries: int,
               wait: int) -> pd.DataFrame:
    """
    Check all URLs in extracted_url column of dataframe

    Parameters:
    `df`: Dataframe with extracted_url column

    Return: Dataframe with extracted_url_status
    and wayback_url columns added
    """

    check_url_part = partial(check_url, num_tries=num_tries, wait=wait)

    with get_pool(cores) as pool:
        logging.debug(
            'Checking extracted URL statuses. '
            'Max attempts: %d. '
            'Time between attempts: %d ms.', num_tries, wait)

        url_statuses = pool.map_async(check_url_part,
                                      df['extracted_url']).get()

        df = merge_url_statuses(df, url_statuses)

        logging.debug('Finished checking extracted URLs.')
        logging.debug('Checking for snapshots of extracted URLs '
                      'on WayBack Machine.')

        df['wayback_url'] = df['extracted_url'].map(check_wayback)

        logging.debug('Finished checking WayBack Machine.')

    return df


# ---------------------------------------------------------------------------
def test_check_urls() -> None:
    """ Test check_urls() """

    in_df = pd.DataFrame(
        [[123, 'Some text', 'https://www.google.com'],
         [456, 'More text', 'http://google.com'],
         [789, 'Foo', 'https://www.amazon.com/afbadfbnvbadfbaefbnaegn']],
        columns=['ID', 'text', 'extracted_url'])

    returned_df = check_urls(in_df, None, 3, 0)
    returned_df.sort_values('ID', inplace=True)

    # Correct number of rows
    assert len(returned_df) == 3

    # Correct columns
    assert all(x == y for x, y in zip(returned_df.columns, [
        'ID', 'text', 'extracted_url', 'extracted_url_status',
        'extracted_url_country', 'extracted_url_latitude',
        'extracted_url_longitude', 'wayback_url'
    ]))


# ---------------------------------------------------------------------------
def regroup_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regroup dataframe to contain one row per article, columns may contain
    list elements

    `df`: Dataframe with one row per URL

    Return: Dataframe with one row per article
    """

    logging.debug('Collapsing columns. One row per article')
    df['extracted_url_status'] = df['extracted_url_status'].astype(str)
    df['extracted_url'] = df['extracted_url'].astype(str)
    df['wayback_url'] = df['wayback_url'].astype(str)

    columns = [
        col for col in df.columns
        if col not in ['extracted_url', 'extracted_url_status', 'wayback_url']
    ]

    def join_commas(x):
        return ', '.join(x)

    out_df = (df.groupby(columns).agg({
        'extracted_url': join_commas,
        'extracted_url_status': join_commas,
        'wayback_url': join_commas
    }).reset_index())

    return out_df


# ---------------------------------------------------------------------------
def test_regroup_df() -> None:
    """ Test regroup_df() """

    in_df = pd.DataFrame(
        [[123, 'Some text', 'https://www.google.com', 200, 'wayback_google'],
         [123, 'Some text', 'http://google.com', 301, 'no_wayback'],
         [
             789, 'Foo', 'https://www.amazon.com/afbadfbnvbadfbaefbnaegn', 404,
             'no_wayback'
         ]],
        columns=[
            'ID', 'text', 'extracted_url', 'extracted_url_status',
            'wayback_url'
        ])

    out_df = pd.DataFrame(
        [[
            123, 'Some text', 'https://www.google.com, http://google.com',
            '200, 301', 'wayback_google, no_wayback'
        ],
         [
             789, 'Foo', 'https://www.amazon.com/afbadfbnvbadfbaefbnaegn',
             '404', 'no_wayback'
         ]],
        columns=[
            'ID', 'text', 'extracted_url', 'extracted_url_status',
            'wayback_url'
        ])

    assert_frame_equal(regroup_df(in_df), out_df)


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()
    out_dir = args.out_dir

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    logging.debug('Reading input file: %s.', args.file.name)
    df = pd.read_csv(args.file)

    df = expand_url_col(df)
    df = check_urls(df, args.cores, args.num_tries, args.wait)
    df = regroup_df(df)

    outfile = make_filename(out_dir, args.file.name)
    df.to_csv(outfile, index=False)

    print(f'Done. Wrote output to {outfile}.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
