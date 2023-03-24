#!/usr/bin/env python3
"""
Purpose: Conduct evaluation on held-out test split
Authors: Kenneth Schackart
"""

import argparse
import os
from typing import BinaryIO, NamedTuple

from torch.utils.data.dataloader import DataLoader

from inventory_utils.custom_classes import CustomHelpFormatter
from inventory_utils.filing import get_ner_model, save_metrics
from inventory_utils.metrics import get_ner_metrics
from inventory_utils.ner_data_handler import RunParams, get_dataloader
from inventory_utils.runtime import get_torch_device


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    test_file: str
    checkpoint: BinaryIO
    out_dir: str


# ---------------------------------------------------------------------------
def get_args():
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Evaluate model on held-out test set',
        formatter_class=CustomHelpFormatter)

    parser.add_argument('-t',
                        '--test-file',
                        metavar='PKL',
                        type=str,
                        required=True,
                        help='Test data file (.pkl)')
    parser.add_argument('-c',
                        '--checkpoint',
                        metavar='PT',
                        type=argparse.FileType('rb'),
                        required=True,
                        help='Trained model checkpoint')
    parser.add_argument('-o',
                        '--out-dir',
                        metavar='DIR',
                        type=str,
                        default='out/',
                        help='Directory to output metrics')

    args = parser.parse_args()

    if ".pkl" not in args.test_file:
        parser.error(f'Invalid input file "{args.test_file}". Must be .pkl')

    return Args(args.test_file, args.checkpoint, args.out_dir)


# ---------------------------------------------------------------------------
def get_test_dataloader(args: Args, model_name: str) -> DataLoader:
    """
    Generate the dataloaders

    Parameters:
    `args`: Command-line arguments
    `model_name`: HuggingFace model name

    Return:
    Test `DataLoader`
    """

    params = RunParams(model_name, 8)

    dataloader = get_dataloader(args.test_file, params)

    return dataloader


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()
    out_dir = args.out_dir

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    out_file = os.path.join(args.out_dir, 'metrics.csv')

    device = get_torch_device()

    model, model_name, _ = get_ner_model(args.checkpoint, device)

    dataloader = get_test_dataloader(args, model_name)

    test_metrics = get_ner_metrics(model, dataloader, device)

    save_metrics(model_name, test_metrics, out_file)

    print(f'Done. Wrote output to {out_dir}.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
