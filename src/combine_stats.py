#!/usr/bin/env python3
"""
Purpose: Combine scores during training of all models
Authors: Kenneth Schackart
"""

import argparse
import os
from typing import List, NamedTuple, TextIO

import pandas as pd

from inventory_utils.custom_classes import CustomHelpFormatter


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    files: List[TextIO]
    out_dir: str


# ---------------------------------------------------------------------------
def get_args() -> Args:
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Combine scores during training of all models',
        formatter_class=CustomHelpFormatter)

    parser.add_argument('files',
                        nargs='+',
                        metavar='FILE',
                        type=argparse.FileType('rt'),
                        help='Model training stats files')
    parser.add_argument('-o',
                        '--out-dir',
                        metavar='DIR',
                        type=str,
                        default='out/',
                        help='Output directory')

    args = parser.parse_args()

    return Args(args.files, args.out_dir)


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    out_df = pd.DataFrame()

    for fh in args.files:
        in_df = pd.read_csv(fh)

        out_df = pd.concat([out_df, in_df])

    out_file = os.path.join(args.out_dir, 'combined_stats.csv')

    out_df.to_csv(out_file, index=False)

    print(f'Done. Wrote output to {out_file}.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
