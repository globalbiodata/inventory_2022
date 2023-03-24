#!/usr/bin/env python3
"""
Purpose: Check URL statuses, attempt to Geolocate, and check WayBack
Authors: Kenneth Schackart
"""

import argparse
import logging
import multiprocessing as mp
import os
import re
import socket
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing.pool import Pool
from typing import List, NamedTuple, Optional, OrderedDict, TextIO, Union, cast

import numpy as np
import pandas as pd
import pytest
import requests
from pandas.testing import assert_frame_equal
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from inventory_utils.custom_classes import CustomHelpFormatter
from inventory_utils.wrangling import chunk_rows, join_commas

# ---------------------------------------------------------------------------
API_REQ_DICT = {
    'ipinfo': 'https://ipinfo.io/{}/json',
    'ip-api': 'http://ip-api.com/json/{}'
}
"""
Dictionary of APIs that geolocate from IP, and their templates.
Fill in template with `API_REQ_DICT[api].format(ip)`

`key`: API name
`value`: Template with `{}` placeholder for IP address
"""


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    file: TextIO
    partial: Optional[TextIO]
    out_dir: str
    verbose: bool
    cores: Optional[int]
    chunk_size: Optional[int]
    num_tries: int
    backoff: float


# ---------------------------------------------------------------------------
class URLStatus(NamedTuple):
    """
    URL and its returned status and location

    `url`: URL string
    `status`: URL status or error message from request
    `country`: Geolocated country from IP address
    `coordinates`: Geolocated coordinates (lan, lon) from IP address
    """
    url: str
    status: Union[str, int]
    country: str
    coordinates: str


# ---------------------------------------------------------------------------
class IPLocation(NamedTuple):
    """
    IP address location

    `country`: Geolocated country from IP address
    `coordinates`: Geolocated coordinates (lan, lon) from IP address
    """
    country: str
    coordinates: str


# ---------------------------------------------------------------------------
@pytest.fixture(name='in_dataframe')
def fixture_in_dataframe() -> requests.Session:
    """ A minimal example of input dataframe, not all columns present """

    in_df = pd.DataFrame([[123, 'Some text', 'http://google.com'],
                          [456, 'More text', 'http://google.com'],
                          [789, 'Third text', 'http://foo.com'],
                          [147, 'Fourth text', 'http://baz.net/']],
                         columns=['ID', 'text', 'extracted_url'])

    return in_df


# ---------------------------------------------------------------------------
def get_args() -> Args:
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description=('Check extracted URL statuses and '
                     'see if snapshot is in WayBack Machine'),
        formatter_class=CustomHelpFormatter)

    inputs = parser.add_argument_group('Inputs and Outputs')
    runtime_params = parser.add_argument_group('Runtime Parameters')
    url_checking = parser.add_argument_group('URL Requesting')

    inputs.add_argument('file',
                        metavar='FILE',
                        type=argparse.FileType('rt', encoding='ISO-8859-1'),
                        help='CSV File with extracted_url column')
    inputs.add_argument('-p',
                        '--partial',
                        metavar='FILE',
                        type=argparse.FileType('rt', encoding='ISO-8859-1'),
                        help='Partially completed output file')
    inputs.add_argument('-o',
                        '--out-dir',
                        metavar='DIR',
                        type=str,
                        default='out/',
                        help='Output directory')

    runtime_params.add_argument('-v',
                                '--verbose',
                                action='store_true',
                                help=('Run with debugging messages'))
    runtime_params.add_argument('-c',
                                '--cores',
                                metavar='CORE',
                                type=int,
                                help=('Number of cores for multi-'
                                      'processing. -c <= 0 will use'
                                      ' -c less than all available. '
                                      'If not supplied, threading is used '
                                      'instead of multiprocessing. '
                                      '(default: threading)'))
    runtime_params.add_argument('-s',
                                '--chunk-size',
                                metavar='INT',
                                type=int,
                                help=('Number of rows '
                                      'to process at a time. Output '
                                      'is appended after each chunk. '
                                      '(default: all)'))

    url_checking.add_argument('-n',
                              '--num-tries',
                              metavar='INT',
                              type=int,
                              default=3,
                              help='Number of tries for checking URL')
    url_checking.add_argument('-b',
                              '--backoff',
                              metavar='[0-1]',
                              type=float,
                              default=0.5,
                              help='Back-off Factor for retries')

    args = parser.parse_args()

    if not 0 <= args.backoff <= 1:
        parser.error(f'--backoff ({args.backoff}) must '
                     'be between 0 and 1, inclusive.')

    if not args.num_tries >= 0:
        parser.error(f'--num-tries ({args.num_tries}) must be at least 1')

    return Args(args.file, args.partial, args.out_dir, args.verbose,
                args.cores, args.chunk_size, args.num_tries, args.backoff)


# ---------------------------------------------------------------------------
def remove_partial(all_df: pd.DataFrame,
                   partial_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows in `all_df` that are in `partial_df`, since their URLs have
    already been checked.

    Parameters:
    `all_df`: Input dataframe containing all rows
    `partial_df`: Dataframe of rows that have been checked

    Return: Dataframe without rows from `partial_df`
    """

    logging.debug('Removing articles present in the partially processed'
                  ' output file.')
    out_df = all_df.copy()
    processed_ids = partial_df['ID']

    logging.debug('Removing %d articles.', len(processed_ids))
    out_df = out_df[~out_df['ID'].isin(processed_ids)]
    out_df.reset_index(inplace=True, drop=True)

    return out_df


# ---------------------------------------------------------------------------
def test_remove_partial(in_dataframe) -> None:
    """ Test remove_partial() """

    part_df = pd.DataFrame(
        [[456, 'More text', 'http://google.com', 200],
         [789, 'Third text', 'http://foo.com', 404]],
        columns=['ID', 'text', 'extracted_url', 'extracted_url_status'])

    out_df = pd.DataFrame([[123, 'Some text', 'http://google.com'],
                           [147, 'Fourth text', 'http://baz.net/']],
                          columns=['ID', 'text', 'extracted_url'])

    assert_frame_equal(remove_partial(in_dataframe, part_df), out_df)


# ---------------------------------------------------------------------------
def get_session(tries: int, backoff: float = 0) -> requests.Session:
    """
    Establish request `Session` applying tries and backoff

    Parameters:
    `tries`: Number of request attempts
    `backoff`: Backoff factor to prevent quota error

    Return: A `requests.Session`
    """

    session = requests.Session()

    # total arg is provided as number of retries, so subtract one
    retry = Retry(total=tries - 1, backoff_factor=backoff)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    return session


# ---------------------------------------------------------------------------
def test_get_session() -> None:
    """ Test get_session() """

    session = get_session(3, 0.5)

    assert isinstance(session, requests.Session)
    assert isinstance(session.adapters, OrderedDict)
    assert isinstance(session.adapters['http://'], HTTPAdapter)
    assert isinstance(session.adapters['http://'].max_retries, Retry)
    assert session.adapters['http://'].max_retries.total == 2


# ---------------------------------------------------------------------------
@pytest.fixture(name='testing_session')
def fixture_testing_session() -> requests.Session:
    """ A basic session used for testing requests """

    return get_session(1, 0)


# ---------------------------------------------------------------------------
def remove_missing_urls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that do not have any URLs

    Parameters:
    `df`: Raw dataframe

    Return: Dataframe with no missing URLs
    """

    return df.replace({
        'extracted_url': ''
    }, np.nan).dropna(subset=['extracted_url'])


# ---------------------------------------------------------------------------
def test_remove_missing_urls() -> None:
    """ Test remove_missing_urls() """

    in_df = pd.DataFrame(
        [[123, 'Some text', 'https://www.google.com, http://google.com'],
         [789, 'Foo', 'https://www.amazon.com/afbadfbnvbadfbaefbnaegn'],
         [147, 'Blah', '']],
        columns=['ID', 'text', 'extracted_url'])

    out_df = pd.DataFrame(
        [[123, 'Some text', 'https://www.google.com, http://google.com'],
         [789, 'Foo', 'https://www.amazon.com/afbadfbnvbadfbaefbnaegn']],
        columns=['ID', 'text', 'extracted_url'])

    assert_frame_equal(remove_missing_urls(in_df), out_df)


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
def get_pool(cores: int) -> Pool:
    """
    Get Pool for multiprocessing.

    Parameters:
    `cores`: Number of cores to use. If `cores` >= 1 , use `cores`.
    If `cores` <= 0, use available cores - |`cores`|

    Return:
    multiprocessing `Pool`
    """

    if cores <= 0:
        n_cores = mp.cpu_count() - abs(cores)
    else:
        n_cores = cores

    logging.debug('Running with %d cores', n_cores)

    return mp.Pool(n_cores)  # pylint: disable=consider-using-with


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
def request_url(url: str, session: requests.Session) -> Union[int, str]:
    """
    Check a URL once using try-except to catch exceptions

    Parameters:
    `url`: URL string
    `session`: request `Session`

    Return: Status code or error message
    """

    try:
        r = session.head(url, timeout=5)
    except requests.exceptions.RequestException as err:
        return str(err)

    return r.status_code


# ---------------------------------------------------------------------------
def test_request_url(testing_session: requests.Session) -> None:
    """ Test request_url() """

    # Hopefully, Google doesn't disappear, if it does use a different URL
    assert request_url('https://www.google.com', testing_session) == 200

    # Bad URLs
    assert int(request_url('http://google.com', testing_session)) >= 300
    assert int(
        request_url('https://www.amazon.com/afbadfbnvbadfbaefbnaegn',
                    testing_session)) >= 300

    # Runtime exception
    assert request_url('adflkbndijfbn', testing_session) == (
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
    Query an API to find location from IP address

    Parameters:
    `ip`: IP address to check
    `api`: API to query

    Return: An `IPLocation` object, which may have empty strings
    """

    logging.debug('Querying %s.', api)
    query_template = API_REQ_DICT[api]

    r = requests.get(query_template.format(ip), verify=True)

    if r.status_code != 200:
        return IPLocation('', '')

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

    coordinates = '(' + latitude + ',' + longitude + ')'
    ip_location = IPLocation(country, coordinates)

    logging.debug('Obtained IP address location: %s', ip_location.country)
    return ip_location


# ---------------------------------------------------------------------------
def get_location(url: str) -> IPLocation:
    """
    Get location of URL by first fetching the IP address of
    the connection, then searching for location of that IP address

    Parameters:
    `url`: URL to search

    Return: An `IPLocation` object, which may have empty strings
    """

    logging.debug('Attempting to determine IP address of %s', url)
    try:
        ip = socket.gethostbyname(extract_domain(url))
    except socket.gaierror:
        ip = ''

    if not ip:
        logging.debug('IP address for %s could not be determined', url)
        return IPLocation('', '')

    logging.debug('IP address found: %s.', ip)
    logging.debug('Attempting to geolocate IP address.')
    location = query_ip(ip, 'ipinfo')
    country = location.country
    coordinates = location.coordinates

    if '' in [country, coordinates]:
        location = query_ip(ip, 'ip-api')

    # Select non-empty location attributes
    country = country if country else location.country
    coordinates = coordinates if coordinates else location.coordinates

    logging.debug('Final location information for %s: %s', ip, country)

    return IPLocation(country, coordinates)


# ---------------------------------------------------------------------------
def test_get_location() -> None:
    """ Test get_location() """

    location = get_location('https://google.com')
    assert location.country != ''

    location = get_location('google.com')
    assert location.country != ''


# ---------------------------------------------------------------------------
def check_url(url: str, session: requests.Session) -> URLStatus:
    """
    Try requesting URL the specified number of tries, returning 200
    if it succeeds at least once

    Parameters:
    `url`: URL string
    `session`: request `Session`

    Return: `URLStatus` object
    """

    location = IPLocation('', '')

    logging.debug('Requesting %s', url)
    status = request_url(url, session)
    logging.debug('Returned status for %s: %s', url, str(status))

    if isinstance(status, int) and status < 400:
        location = get_location(url)

    return URLStatus(url, status, location.country, location.coordinates)


@pytest.mark.slow
# ---------------------------------------------------------------------------
def test_check_url(testing_session: requests.Session) -> None:
    """ Test check_url() """

    url_status = check_url('https://www.google.com', testing_session)
    assert url_status.url == 'https://www.google.com'
    assert url_status.status == 200

    # Bad URLs
    url_status = check_url('http://google.com', testing_session)
    assert url_status.url == 'http://google.com'
    assert isinstance(url_status.status, int)
    assert url_status.status >= 300

    url_status = check_url('https://www.amazon.com/afbadffbaefbnaegn',
                           testing_session)
    assert url_status.url == 'https://www.amazon.com/afbadffbaefbnaegn'
    assert isinstance(url_status.status, int)
    assert url_status.status >= 400
    assert url_status.country == ''

    # Runtime exception
    url_status = check_url('adflkbndijfbn', testing_session)
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
            'coordinates': x.coordinates
        }
        for x in url_statuses
    }

    df['extracted_url_status'] = df['extracted_url'].map(
        lambda x: url_dict[x]['status'])
    df['extracted_url_country'] = df['extracted_url'].map(
        lambda x: url_dict[x]['country'])
    df['extracted_url_coordinates'] = df['extracted_url'].map(
        lambda x: url_dict[x]['coordinates'])

    return df


# ---------------------------------------------------------------------------
def test_merge_url_statuses() -> None:
    """ Test merge_url_statuses() """

    in_df = pd.DataFrame([[123, 'Some text', 'https://www.google.com'],
                          [456, 'More text', 'http://google.com']],
                         columns=['ID', 'text', 'extracted_url'])

    statuses = [
        URLStatus('http://google.com', 301, '', ''),
        URLStatus('https://www.google.com', 200, 'United States',
                  '(34.0522,-118.2437)')
    ]

    out_df = pd.DataFrame([[
        123, 'Some text', 'https://www.google.com', 200, 'United States',
        '(34.0522,-118.2437)'
    ], [456, 'More text', 'http://google.com', 301, '', '']],
                          columns=[
                              'ID', 'text', 'extracted_url',
                              'extracted_url_status', 'extracted_url_country',
                              'extracted_url_coordinates'
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

    # Not try-except because if there is an exception it is not
    # because there is not an archived version, it means the API
    # has changed. The code must be updated then.
    r = requests.get(f'http://archive.org/wayback/available?url={url}',
                     headers={'User-agent': 'biodata_resource_inventory'})

    if r.status_code in [504, 503]:
        print('WARNING: WayBack Machine is not responding')
        return 'wayback is down'

    returned_dict = cast(dict, r.json())
    snapshots = cast(dict, returned_dict.get('archived_snapshots'))

    if not snapshots:
        return 'no_wayback'

    snapshot = cast(dict, snapshots.get('closest'))

    return snapshot.get('url', 'no_wayback')


@pytest.mark.slow
# ---------------------------------------------------------------------------
def test_check_wayback() -> None:
    """ Test check_wayback() """

    # Example from their website
    assert check_wayback('example.com') != ''

    # Valid URL, but not present as a snapshot

    # Invalid URL
    assert check_wayback('aegkbnwefnb') == 'no_wayback'


# ---------------------------------------------------------------------------
def check_urls(df: pd.DataFrame, cores: Optional[int],
               session: requests.Session) -> pd.DataFrame:
    """
    Check all URLs in extracted_url column of dataframe

    Parameters:
    `df`: Dataframe with extracted_url column
    `cores`: (optional) number of cores to use
    `session`: requests `Session`

    Return: Dataframe with extracted_url_status
    and wayback_url columns added
    """

    check_url_part = partial(check_url, session=session)
    out_df = df.copy()

    if cores is not None:
        logging.debug('Starting multiple processes.')
        with get_pool(cores) as pool:
            logging.debug('Checking extracted URL statuses. ')

            url_statuses = pool.map_async(check_url_part,
                                          out_df['extracted_url']).get()
    else:
        logging.debug('Starting thread pool.')
        with ThreadPoolExecutor() as executor:
            logging.debug('Checking extracted URL statuses. ')

            url_statuses = list(
                executor.map(check_url_part, out_df['extracted_url']))

    out_df = merge_url_statuses(out_df, url_statuses)

    logging.debug('Finished checking extracted URLs.')
    logging.debug('Checking for snapshots of extracted URLs '
                  'on WayBack Machine.')

    out_df['wayback_url'] = out_df['extracted_url'].map(check_wayback)

    logging.debug('Finished checking WayBack Machine.')

    return out_df


@pytest.mark.slow
# ---------------------------------------------------------------------------
def test_check_urls(testing_session: requests.Session) -> None:
    """ Test check_urls() """

    in_df = pd.DataFrame(
        [[123, 'Some text', 'https://www.google.com'],
         [456, 'More text', 'http://google.com'],
         [789, 'Foo', 'https://www.amazon.com/afbadfbnvbadfbaefbnaegn']],
        columns=['ID', 'text', 'extracted_url'])

    # Check that it can run with either threading or multiprocessing
    # cores == None -> threading
    # cores == 0 or -1 -> multiprocessing
    for cores in [None, 0, -1]:
        returned_df = check_urls(in_df, cores, testing_session)

        # Correct number of rows
        assert len(returned_df) == 3

        # Correct columns
        assert (returned_df.columns == [
            'ID', 'text', 'extracted_url', 'extracted_url_status',
            'extracted_url_country', 'extracted_url_coordinates', 'wayback_url'
        ]).all()


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

    out_df = (df.groupby(['ID']).agg({
        'best_name': 'first',
        'best_name_prob': 'first',
        'best_common': 'first',
        'best_common_prob': 'first',
        'best_full': 'first',
        'best_full_prob': 'first',
        'article_count': 'first',
        'publication_date': 'first',
        'extracted_url': join_commas,
        'extracted_url_status': join_commas,
        'extracted_url_country': join_commas,
        'extracted_url_coordinates': join_commas,
        'wayback_url': join_commas,
    }).reset_index())

    return out_df


# ---------------------------------------------------------------------------
def test_regroup_df() -> None:
    """ Test regroup_df() """

    in_df = pd.DataFrame(
        [[
            123, 'Some text', 'google', 0.99, 'google', 0.99, '', '',
            'https://www.google.com', 200, 'US', '(12,10)', 'wayback_google',
            '2012-01-01', '2'
        ],
         [
             123, 'Some text', 'google', 0.99, 'google', 0.99, '', '',
             'http://google.com', 301, 'US', '(100,17)', 'no_wayback',
             '2012-01-01', '2'
         ],
         [
             789, 'Foo', 'amazon', 0.87, 'amazon', 0.87, 'The Amazon', 0.65,
             'https://www.amazon.com/afbadfbnvbadfbaefbnaegn', 404, '', '',
             'no_wayback', '2012-01-02', '1'
         ]],
        columns=[
            'ID', 'text', 'best_name', 'best_name_prob', 'best_common',
            'best_common_prob', 'best_full', 'best_full_prob', 'extracted_url',
            'extracted_url_status', 'extracted_url_country',
            'extracted_url_coordinates', 'wayback_url', 'publication_date',
            'article_count'
        ])

    out_df = pd.DataFrame(
        [[
            123, 'google', 0.99, 'google', 0.99, '', '', '2', '2012-01-01',
            'https://www.google.com, http://google.com', '200, 301', 'US, US',
            '(12,10), (100,17)', 'wayback_google, no_wayback'
        ],
         [
             789, 'amazon', 0.87, 'amazon', 0.87, 'The Amazon', 0.65, '1',
             '2012-01-02', 'https://www.amazon.com/afbadfbnvbadfbaefbnaegn',
             '404', '', '', 'no_wayback'
         ]],
        columns=[
            'ID', 'best_name', 'best_name_prob', 'best_common',
            'best_common_prob', 'best_full', 'best_full_prob', 'article_count',
            'publication_date', 'extracted_url', 'extracted_url_status',
            'extracted_url_country', 'extracted_url_coordinates', 'wayback_url'
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

    outfile = make_filename(out_dir, args.file.name)

    session = get_session(args.num_tries, args.backoff)

    logging.debug('Reading input file: %s.', args.file.name)
    df = remove_missing_urls(pd.read_csv(args.file, dtype=str))

    if args.partial:
        part_df = pd.read_csv(args.partial, dtype=str)
        df = remove_partial(df, part_df)
        part_df.to_csv(outfile, index=False)

    for i, chunk in enumerate(chunk_rows(df, args.chunk_size)):
        logging.debug('Processing chunk %d (%d articles).', i + 1, len(chunk))
        df = expand_url_col(chunk)
        df = check_urls(df, args.cores, session)
        df = regroup_df(df)

        logging.debug('Writing intermediate output to %s.', outfile)
        df.to_csv(outfile,
                  index=False,
                  mode='a',
                  header=not os.path.exists(outfile))

    session.close()

    print(f'Done. Wrote output to {outfile}.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
