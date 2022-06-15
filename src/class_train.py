#!/usr/bin/env python3
"""
Purpose: Train BERT model for article classification
Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import argparse
import copy
import os
from typing import Any, List, NamedTuple, TextIO, Tuple, cast

import pandas as pd
import torch
from datasets import load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import (AdamW, AutoModelForSequenceClassification,
                          get_scheduler)

from class_data_handler import DataFields, RunParams, get_dataloader
from utils import (CustomHelpFormatter, Metrics, Settings, make_filenames,
                   save_model, save_train_stats, set_random_seed)


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    train_file: TextIO
    val_file: TextIO
    out_dir: str
    predictive_field: str
    labels_field: str
    descriptive_labels: List[str]
    model_name: str
    max_len: int
    learning_rate: float
    weight_decay: float
    num_training: int
    num_epochs: int
    batch_size: int
    lr_scheduler: bool
    seed: bool


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
                        '--train-file',
                        metavar='FILE',
                        type=argparse.FileType('rt'),
                        default='data/train.csv',
                        help='Training data file')
    inputs.add_argument('-v',
                        '--val-file',
                        metavar='FILE',
                        type=argparse.FileType('rt'),
                        default='data/val.csv',
                        help='Validation data file')
    inputs.add_argument('-o',
                        '--out-dir',
                        metavar='DIR',
                        type=str,
                        default='out/',
                        help='Directory to output checkpt and loss plot')

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

    model_params.add_argument('-m',
                              '--model-name',
                              metavar='MODEL',
                              type=str,
                              required=True,
                              help='Name of model')
    model_params.add_argument('-max',
                              '--max-len',
                              metavar='INT',
                              type=int,
                              default=256,
                              help='Max Sequence Length')
    model_params.add_argument('-rate',
                              '--learning-rate',
                              metavar='NUM',
                              type=float,
                              default=2e-5,
                              help='Learning Rate')
    model_params.add_argument('-decay',
                              '--weight-decay',
                              metavar='NUM',
                              type=float,
                              default=0.0,
                              help='Weight Decay for Learning Rate')

    runtime_params.add_argument(
        '-nt',
        '--num-training',
        metavar='INT',
        type=int,
        default=None,
        help='Number of data points for training (default: all)')
    runtime_params.add_argument('-ne',
                                '--num-epochs',
                                metavar='INT',
                                type=int,
                                default=10,
                                help='Number of Epochs')
    runtime_params.add_argument('-batch',
                                '--batch-size',
                                metavar='INT',
                                type=int,
                                default=32,
                                help='Batch Size')
    runtime_params.add_argument('-lr',
                                '--lr-scheduler',
                                action='store_true',
                                help='Use a Learning Rate Scheduler')
    runtime_params.add_argument('-r',
                                '--seed',
                                action='store_true',
                                help='Set random seed')

    args = parser.parse_args()

    return Args(args.train_file, args.val_file, args.out_dir,
                args.predictive_field, args.labels_field,
                args.descriptive_labels, args.model_name, args.max_len,
                args.learning_rate, args.weight_decay, args.num_training,
                args.num_epochs, args.batch_size, args.lr_scheduler, args.seed)


# ---------------------------------------------------------------------------
def train(settings: Settings) -> Tuple[Any, pd.DataFrame]:
    """
    Train the classifier

    Parameters:
    `settings`: Model settings (NamedTuple)

    Return: Tuple of best model, and training stats dataframe
    """

    model = settings.model
    progress_bar = tqdm(range(settings.num_training_steps))
    train_progress = pd.DataFrame(columns=[
        'epoch', 'train_precision', 'train_recall', 'train_f1', 'train_loss',
        'val_precision', 'val_recall', 'val_f1', 'val_loss'
    ])
    best_model = model
    best_val = Metrics(0, 0, 0, 0)
    best_train = Metrics(0, 0, 0, 0)
    model.train()

    for epoch in range(settings.num_epochs):

        train_loss = train_epoch(settings, progress_bar)

        model.eval()
        train_metrics = get_metrics(model, settings.train_dataloader,
                                    settings.device)
        val_metrics = get_metrics(model, settings.val_dataloader,
                                  settings.device)

        if val_metrics.f1 > best_val.f1:
            best_val = val_metrics
            best_train = train_metrics
            best_model = copy.deepcopy(model)

        # Stop training once validation F1 goes down
        # Overfitting has begun
        # if val_metrics.f1 < best_val.f1 and epoch > 0:
        #     break

        epoch_row = pd.DataFrame(
            {
                'epoch': epoch,
                'train_precision': train_metrics.precision,
                'train_recall': train_metrics.recall,
                'train_f1': train_metrics.f1,
                'train_loss': train_metrics.loss,
                'val_precision': val_metrics.precision,
                'val_recall': val_metrics.recall,
                'val_f1': val_metrics.f1,
                'val_loss': val_metrics.loss
            },
            index=[0])
        train_progress = pd.concat([train_progress, epoch_row])

        print(f'Epoch {epoch + 1}:\n'
              f'Train Loss: {train_loss:.5f}\n'
              f'Val Loss: {val_metrics.loss:.5f}\n'
              f'Train Precision: {train_metrics.precision:.3f}\n'
              f'Train Recall: {train_metrics.recall:.3f}\n'
              f'Train F1: {train_metrics.f1:.3f}\n'
              f'Val Precision: {val_metrics.precision:.3f}\n'
              f'Val Recall: {val_metrics.recall:.3f}\n'
              f'Val F1: {val_metrics.f1:.3f}')

    print('Finished model training!')
    print('=' * 30)
    print(f'Best Train Precision: {best_train.precision:.3f}\n'
          f'Best Train Recall: {best_train.recall:.3f}\n'
          f'Best Train F1: {best_train.f1:.3f}\n'
          f'Best Val Precision: {best_val.precision:.3f}\n'
          f'Best Val Recall: {best_val.recall:.3f}\n'
          f'Best Val F1: {best_val.f1:.3f}\n')

    return best_model, train_progress


# ---------------------------------------------------------------------------
def train_epoch(settings: Settings, progress_bar: tqdm) -> float:
    """
    Perform one epoch of model training

    Parameters:
    `settings`: Model settings (NamedTuple)
    `progress_bar`: tqdm instance for tracking progress

    Return: Average train loss per observation
    """
    train_loss = 0
    num_train = 0
    for batch in settings.train_dataloader:
        batch = {k: v.to(settings.device) for k, v in batch.items()}
        num_train += len(batch['input_ids'])
        outputs = settings.model(**batch)
        loss = outputs.loss
        loss.backward()
        train_loss += loss.item()
        settings.optimizer.step()
        if settings.lr_scheduler:
            settings.lr_scheduler.step()
        settings.optimizer.zero_grad()
        progress_bar.update(1)
    return train_loss / num_train


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
def get_dataloaders(args: Args,
                    model_name: str) -> Tuple[DataLoader, DataLoader]:
    """
    Generate the dataloaders

    Parameters:
    `args`: Command-line arguments
    `model_name`: Huggingface model name

    Return:
    A Tuple of trianing, validation `DataLoader`s
    """

    print('Generating dataloaders ...')
    print('=' * 30)

    data_fields = DataFields(
        args.predictive_field,
        args.descriptive_labels,
        args.labels_field,
    )

    dataloader_params = RunParams(model_name, args.batch_size, args.max_len,
                                  args.num_training)

    train_dataloader = get_dataloader(args.train_file, data_fields,
                                      dataloader_params)
    val_dataloader = get_dataloader(args.val_file, data_fields,
                                    dataloader_params)

    print('Finished generating dataloaders!')
    print('=' * 30)

    return train_dataloader, val_dataloader


# ---------------------------------------------------------------------------
def initialize_model(model_name: str, args: Args, train_dataloader: DataLoader,
                     val_dataloader: DataLoader) -> Settings:
    """
    Instatiate predictive model from HFHub and get settings

    Params:
    `model_name`: Pretrained model name
    `args`: Command-line arguments
    `trin_dataloader`: Training `DataLoader`
    `val_dataloader`: Validation `DataLoader`

    Return:
    `Settings` including pretrained model
    """

    print(f'Initializing {model_name} model ...')
    print('=' * 30)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               num_labels=2)
    optimizer = AdamW(model.parameters(),
                      lr=args.learning_rate,
                      weight_decay=args.weight_decay)
    num_training_steps = args.num_epochs * len(train_dataloader)
    if args.lr_scheduler:
        lr_scheduler = get_scheduler('linear',
                                     optimizer=optimizer,
                                     num_warmup_steps=0,
                                     num_training_steps=num_training_steps)
    else:
        lr_scheduler = None
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    return Settings(model, optimizer, train_dataloader, val_dataloader,
                    lr_scheduler, args.num_epochs, num_training_steps, device)


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()
    out_dir = args.out_dir

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    model_name = args.model_name

    train_dataloader, val_dataloader = get_dataloaders(args, model_name)

    if args.seed:
        set_random_seed(45)
    settings = initialize_model(model_name, args, train_dataloader,
                                val_dataloader)

    print('Starting model training...')
    print('=' * 30)

    model, train_stats_df = train(settings)
    train_stats_df['model_name'] = model_name

    checkpt_filename, train_stats_filename = make_filenames(out_dir)
    save_model(model, model_name, checkpt_filename)
    save_train_stats(train_stats_df, train_stats_filename)

    print('Done. Saved best checkpoint to', checkpt_filename)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
