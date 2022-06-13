#!/usr/bin/env python3
"""
Purpose: Choose model based on highest validation F1 score
Authors: Kenneth Schackart
"""

import argparse
import os
import re
import string
from typing import List, NamedTuple, Set, TextIO

import pandas as pd
from pandas.testing import assert_frame_equal

from utils import CustomHelpFormatter, preprocess_data


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    file: TextIO
    out_dir: str


# ---------------------------------------------------------------------------
def get_args() -> Args:
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Extract URLs from title and abstracts',
        formatter_class=CustomHelpFormatter)

    parser.add_argument('file',
                        metavar='FILE',
                        type=argparse.FileType('rt', encoding='ISO-8859-1'),
                        help='Input file')
    parser.add_argument('-o',
                        '--out-dir',
                        metavar='DIR',
                        type=str,
                        default='out/',
                        help='Output directory')

    args = parser.parse_args()

    return Args(args.file, args.out_dir)


# ---------------------------------------------------------------------------
def extract_urls(text: str) -> List[str]:
    """ Extract URLs from a string """

    url_pattern = re.compile(
        r'''http[s]? # http and optional s
         :// # Literal ://
         (?:[\w$-_@.&+!*\(\),] # Any word or number chars or these symbols
         )+''', re.X)

    urls = re.findall(url_pattern, text)
    bad_punct = re.sub('/', '', string.punctuation)  # Do not remove trailing /
    urls = list(map(lambda s: s.strip(bad_punct), urls))

    # Remove duplicates
    seen: Set[str] = set()
    seen_add = seen.add
    urls = [x for x in urls if not (x in seen or seen_add(x))]

    return urls


# ---------------------------------------------------------------------------
def test_extract_urls() -> None:
    """ Test extract_urls() """

    # Single URL
    in_str = 'http://bacdb.org/BacWGSTdb/'
    out = ['http://bacdb.org/BacWGSTdb/']
    assert extract_urls(in_str) == out

    # Multiple URLs
    in_str = 'http://mirtrondb.cp.utfpr.edu.br/ http://bbcancer.renlab.org/'
    out = ['http://mirtrondb.cp.utfpr.edu.br/', 'http://bbcancer.renlab.org/']
    assert extract_urls(in_str) == out

    # No extraneous words
    in_str = 'Extraneous http://AciDB.cl words!'
    out = ['http://AciDB.cl']
    assert extract_urls(in_str) == out

    # Various formats seen
    in_str = (
        'https://exobcd.liumwei.org '  # https
        'https://enset-project.org/EnMom@base.html '  # @ sign
        'http://oka.protres.ru:4200 '  # colon later in string
        '(https://gitlab.pasteur.fr/hub/viralhostrangedb). '  # Parens
        'http://evpedia.info http://evpedia.info '  # Duplicates
    )
    out = [
        'https://exobcd.liumwei.org',
        'https://enset-project.org/EnMom@base.html',
        'http://oka.protres.ru:4200',
        'https://gitlab.pasteur.fr/hub/viralhostrangedb', 'http://evpedia.info'
    ]
    assert extract_urls(in_str) == out


# ---------------------------------------------------------------------------
def add_url_column(df: pd.DataFrame) -> pd.DataFrame:
    """ Add column of extracted URLs"""

    df['url'] = df['text'].apply(extract_urls)

    df['url'] = df['url'].apply(', '.join)

    return df


# ---------------------------------------------------------------------------
def test_add_url_column() -> None:
    """ Test add_url_column """

    in_df = pd.DataFrame(
        [['123', 'ATAV (http://atavdb.org)', 'ATAV', '0.995', '', ''],
         [
             '456',
             'https://pharos.nih.gov/ and http://juniper.health.unm.edu/tcrd/',
             'Pharos', '0.961', '', ''
         ]],
        columns=[
            'ID', 'text', 'common_name', 'common_prob', 'full_name',
            'full_prob'
        ])

    out_df = pd.DataFrame(
        [[
            '123', 'ATAV (http://atavdb.org)', 'ATAV', '0.995', '', '',
            'http://atavdb.org'
        ],
         [
             '456',
             'https://pharos.nih.gov/ and http://juniper.health.unm.edu/tcrd/',
             'Pharos', '0.961', '', '',
             'https://pharos.nih.gov/, http://juniper.health.unm.edu/tcrd/'
         ]],
        columns=[
            'ID', 'text', 'common_name', 'common_prob', 'full_name',
            'full_prob', 'url'
        ])

    assert_frame_equal(add_url_column(in_df), out_df)


# ---------------------------------------------------------------------------
def get_outname(outdir: str, filename: str) -> str:
    """ Creaate output file name """

    return os.path.join(outdir, os.path.basename(filename))


# ---------------------------------------------------------------------------
def test_get_outname() -> None:
    """ Test get_outname() """

    assert get_outname(
        'out', 'data/ner_predict/predictions.csv') == 'out/predictions.csv'


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()
    out_dir = args.out_dir

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    df = pd.read_csv(args.file)

    if 'text' not in df.columns:
        df = preprocess_data(df)
        df = df.rename(columns={'title_abstract': 'text'})

    df = add_url_column(df)

    out_name = get_outname(out_dir, args.file.name)

    df.to_csv(out_name)

    print(f'Done. Wrote output to {out_name}.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
