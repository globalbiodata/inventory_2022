#!/usr/bin/env python3
"""
Purpose: Conduct evaluation on held-out test split
Authors: Kenneth Schackart
"""

import argparse
import os
from typing import BinaryIO, List, NamedTuple, TextIO

from torch.utils.data.dataloader import DataLoader

from class_data_handler import DataFields, RunParams, get_dataloader
from utils import (CustomHelpFormatter, get_classif_metrics, get_classif_model,
                   get_torch_device, save_metrics)


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    test_file: TextIO
    checkpoint: BinaryIO
    out_dir: str
    predictive_field: str
    descriptive_labels: List[str]
    labels_field: str
    max_len: int
    batch_size: int


# ---------------------------------------------------------------------------
def get_args():
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Evaluate model on held-out test set',
        formatter_class=CustomHelpFormatter)

    inputs = parser.add_argument_group('Inputs and Outputs')
    data_info = parser.add_argument_group('Information on Data')
    model_params = parser.add_argument_group('Model Parameters')
    runtime_params = parser.add_argument_group('Runtime Parameters')

    inputs.add_argument('-t',
                        '--test-file',
                        metavar='FILE',
                        type=argparse.FileType('rt'),
                        required=True,
                        help='Test data file')
    inputs.add_argument('-c',
                        '--checkpoint',
                        metavar='PT',
                        type=argparse.FileType('rb'),
                        required=True,
                        help='Trained model checkpoint')
    inputs.add_argument('-o',
                        '--out-dir',
                        metavar='DIR',
                        type=str,
                        default='out/',
                        required=True,
                        help='Directory to output metrics')

    data_info.add_argument('-pred',
                           '--predictive-field',
                           metavar='PRED',
                           type=str,
                           default='title_abstract',
                           help='Data column to use for prediction',
                           choices=['title', 'abstract', 'title_abstract'])
    data_info.add_argument('-labs',
                           '--labels-field',
                           metavar='LABS',
                           type=str,
                           default='curation_score',
                           help='Data column with classification labels')
    data_info.add_argument('-desc',
                           '--descriptive-labels',
                           metavar='LAB',
                           type=str,
                           nargs=2,
                           default=['not-bio-resource', 'bio-resource'],
                           help='Descriptions of the classification labels')

    model_params.add_argument('-max',
                              '--max-len',
                              metavar='INT',
                              type=int,
                              default=256,
                              help='Max Sequence Length')

    runtime_params.add_argument('-batch',
                                '--batch-size',
                                metavar='INT',
                                type=int,
                                default=8,
                                help='Batch Size')

    args = parser.parse_args()

    return Args(args.test_file, args.checkpoint, args.out_dir,
                args.predictive_field, args.descriptive_labels,
                args.labels_field, args.max_len, args.batch_size)


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

    data_fields = DataFields(
        args.predictive_field,
        args.descriptive_labels,
        args.labels_field,
    )

    dataloader_params = RunParams(model_name, args.batch_size, args.max_len)

    dataloader = get_dataloader(args.test_file, data_fields, dataloader_params)

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

    model, model_name = get_classif_model(args.checkpoint, device)

    dataloader = get_test_dataloader(args, model_name)

    test_metrics = get_classif_metrics(model, dataloader, device)

    save_metrics(test_metrics, out_file)

    print(f'Done. Wrote output to {out_dir}.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
