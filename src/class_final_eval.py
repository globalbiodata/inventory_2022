#!/usr/bin/env python3
"""
Purpose: Conduct evaluation on held-out test split
Authors: Kenneth Schackart
"""

import argparse
import os
from typing import Any, BinaryIO, List, NamedTuple, TextIO, Tuple, cast

import pandas as pd
import torch
from datasets import load_metric
from torch.utils.data.dataloader import DataLoader
from transformers import AutoModelForSequenceClassification as classifier

from class_data_handler import DataFields, RunParams, get_dataloader
from utils import CustomHelpFormatter, Metrics, get_torch_device


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    test_file: TextIO
    checkpoint: BinaryIO
    out_dir: str
    predictive_field: str
    labels_field: str
    descriptive_labels: List[str]
    max_len: int
    batch_size: int


# ---------------------------------------------------------------------------
def get_args():
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Train BERT model for article classification',
        formatter_class=CustomHelpFormatter)

    inputs = parser.add_argument_group('Inputs and Outputs')
    data_info = parser.add_argument_group('Information on Data')
    model_params = parser.add_argument_group('Model Parameters')
    runtime_params = parser.add_argument_group('Runtime Parameters')

    inputs.add_argument('-t',
                        '--test_file',
                        metavar='FILE',
                        type=argparse.FileType('rt'),
                        required=True,
                        help='Test data file')
    inputs.add_argument('-c',
                        '--checkpoint',
                        metavar='',
                        type=argparse.FileType('rb'),
                        required=True,
                        help='Trained model checkpoint')
    inputs.add_argument('-o',
                        '--out-dir',
                        metavar='DIR',
                        type=str,
                        default='out/',
                        help='Directory to output prediction and metrics')

    data_info.add_argument('-pred',
                           '--predictive-field',
                           metavar='',
                           type=str,
                           default='title_abstract',
                           help='Data column to use for prediction',
                           choices=['title', 'abstract', 'title_abstract'])
    data_info.add_argument('-labs',
                           '--labels-field',
                           metavar='',
                           type=str,
                           default='curation_score',
                           help='Data column with classification labels')

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

    return Args(args.test_file, args.checkpoint, args.out_dir, args.checkpoint,
                args.predictive_field, args.descriptive_label, args.max_len,
                args.batch_size)


# ---------------------------------------------------------------------------
def get_model(checkpoint_fh: BinaryIO,
              device: torch.device) -> Tuple[Any, str]:
    """
    Instatiate predictive model from checkpoint

    Params:
    `checkpoint_fh`: Model checkpoint filehandle
    `device`: The `torch.device` to use

    Return:
    Model instance from checkpoint, and model name
    """

    checkpoint = torch.load(checkpoint_fh, map_location=device)
    model_name = checkpoint['model_name']
    model = classifier.from_pretrained(model_name, num_labels=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, model_name


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
def predict(model, dataloader: DataLoader, device: torch.device) -> List[int]:
    """
    Use model to predict article classifications

    Parameters:
    `model`: Pretrained predictive model
    `dataloader`: `DataLoader` with preprocessed data
    `device`: The `torch.device` to use

    Return:
    List of predicted labels
    """

    all_predictions = []
    model.eval()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        all_predictions.extend(predictions)

    return all_predictions


# ---------------------------------------------------------------------------
def get_metrics(model: Any, dataloader: DataLoader,
                device: torch.device) -> Metrics:
    """
    Compute model performance metrics

    Parameters:
    `model`: Classification model
    `dataloader`: DataLoader containing tokenized text entries and
    corresponding labels
    `device`: Torch device

    Return:
    A `Metrics` NamedTuple
    """
    calc_precision = load_metric('precision')
    calc_recall = load_metric('recall')
    calc_f1 = load_metric('f1')
    total_loss = 0.
    num_seen_datapoints = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        num_seen_datapoints += len(batch['input_ids'])
        predictions = torch.argmax(outputs.logits, dim=-1)
        calc_precision.add_batch(predictions=predictions,
                                 references=batch['labels'])
        calc_recall.add_batch(predictions=predictions,
                              references=batch['labels'])
        calc_f1.add_batch(predictions=predictions, references=batch['labels'])
        total_loss += outputs.loss.item()
    total_loss /= num_seen_datapoints

    precision = cast(dict, calc_precision.compute())
    recall = cast(dict, calc_recall.compute())
    f1 = cast(dict, calc_f1.compute())

    return Metrics(precision['precision'], recall['recall'], f1['f1'],
                   total_loss)


# ---------------------------------------------------------------------------
def save_metrics(metrics: Metrics, filename: str) -> None:
    """
    Save test metrics to text file

    Parameters:
    `metrics`: A `Metrics` NamedTuple
    """

    with open(filename, 'wt') as fh:
        print('precision,recall,f1,loss', file=fh)
        print(f'{metrics.precision},{metrics.recall},',
              f'{metrics.f1},{metrics.loss}',
              sep='',
              file=fh)


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()
    out_dir = args.out_dir

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    pred_out_file = os.path.join(args.out_dir, 'predictions.csv')
    metrics_out_file = os.path.join(args.out_dir, 'metrics.csv')

    device = get_torch_device()

    model, model_name = get_model(args.checkpoint, device)

    dataloader = get_test_dataloader(args, model_name)

    predicted_labels = predict(model, dataloader, device)
    df = pd.read_csv(open(args.test_file.name, encoding='ISO-8859-1'))
    df['predicted_label'] = predicted_labels

    test_metrics = get_metrics(model, dataloader, device)

    df = df.replace(r'\n', ' ', regex=True)
    df.to_csv(pred_out_file, index=False)

    save_metrics(test_metrics, metrics_out_file)

    print(f'Done. Wrote 2 files to {out_dir}.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
