#!/usr/bin/env python3
"""
Purpose: Train NER model from pretrained BERT
Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import argparse
import copy
import os
from typing import Any, NamedTuple, Optional, Tuple, cast

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoModelForTokenClassification, get_scheduler,
                          optimization)

from ner_data_handler import RunParams, get_dataloader
from utils import (ID2NER_TAG, NER_TAG2ID, CustomHelpFormatter, Metrics,
                   Settings, get_ner_metrics, make_filenames, save_model,
                   save_train_stats, set_random_seed)


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    train_file: str
    val_file: str
    out_dir: str
    model_name: str
    learning_rate: float
    weight_decay: float
    num_training: int
    num_epochs: int
    batch_size: int
    lr_scheduler: bool
    model_checkpoint: Optional[str]
    seed: bool


# ---------------------------------------------------------------------------
def get_args() -> Args:
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Train BERT model for named entity recognition',
        formatter_class=CustomHelpFormatter)

    inputs = parser.add_argument_group('Inputs and Outputs')
    model_params = parser.add_argument_group('Model Parameters')
    runtime_params = parser.add_argument_group('Runtime Parameters')

    inputs.add_argument('-t',
                        '--train_file',
                        metavar='FILE',
                        type=str,
                        default='data/ner_train.pkl',
                        help='Training data file (.pkl)')
    inputs.add_argument('-v',
                        '--val_file',
                        metavar='FILE',
                        type=str,
                        default='data/ner_val.pkl',
                        help='Validation data file (.pkl)')
    inputs.add_argument('-o',
                        '--out-dir',
                        metavar='DIR',
                        type=str,
                        default='out/',
                        help='Directory to output checkpt and loss plot')

    model_params.add_argument('-m',
                              '--model_name',
                              metavar='',
                              type=str,
                              required=True,
                              help='Name of HuggingFace model')
    model_params.add_argument('-rate',
                              '--learning-rate',
                              metavar='NUM',
                              type=float,
                              default=3e-5,
                              help='Learning rate')
    model_params.add_argument('-decay',
                              '--weight_decay',
                              metavar='NUM',
                              type=float,
                              default=0.01,
                              help='Weight decay for learning rate')

    runtime_params.add_argument(
        '-nt',
        '--num-training',
        metavar='INT',
        type=int,
        default=None,
        help='Number of data points for training (default: all)')
    runtime_params.add_argument('-ne',
                                '--num_epochs',
                                metavar='INT',
                                type=int,
                                default=3,
                                help='Number of epochs')
    runtime_params.add_argument('-batch',
                                '--batch-size',
                                metavar='INT',
                                type=int,
                                default=16,
                                help='Batch size')
    runtime_params.add_argument('-lr',
                                '--lr_scheduler',
                                action='store_true',
                                help='Use a learning rate scheduler')
    runtime_params.add_argument('-r',
                                '--seed',
                                action='store_true',
                                help='Set random seed')

    args = parser.parse_args()

    return Args(args.train_file, args.val_file, args.out_dir, args.model_name,
                args.learning_rate, args.weight_decay, args.num_training,
                args.num_epochs, args.batch_size, args.lr_scheduler, None,
                args.seed)


# ---------------------------------------------------------------------------
def get_dataloaders(args) -> Tuple[DataLoader, DataLoader]:
    """
    Generate training and validation dataloaders

    Parameters:
    `args`: Command-line arguments

    Return: training dataloader, validation dataloader
    """

    print('Generating training and validation dataloaders ...')
    print('=' * 30)

    params = RunParams(args.model_name, args.batch_size, args.num_training)
    train_dataloader = get_dataloader(args.train_file, params)
    val_dataloader = get_dataloader(args.val_file, params)

    print('Finished generating dataloaders!')
    print('=' * 30)

    return train_dataloader, val_dataloader


# ---------------------------------------------------------------------------
def initialize_model(args: Args, train_dataloader: DataLoader,
                     val_dataloader: DataLoader) -> Settings:
    """
    Initialize the model and get settings

    `args`: Command-line arguments
    `train_dataloader`: Training dataloader
    `val_dataloader`: Validation dataloader

    Return: Training settings including model
    """

    print('Initializing', args.model_name, 'model ...')
    print('=' * 30)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, id2label=ID2NER_TAG, label2id=NER_TAG2ID)
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    optimizer = cast(
        optimization.AdamW,
        AdamW(model.parameters(),
              lr=args.learning_rate,
              weight_decay=args.weight_decay))
    num_training_steps = args.num_epochs * len(train_dataloader)

    if args.lr_scheduler:
        lr_scheduler = get_scheduler('linear',
                                     optimizer=optimizer,
                                     num_warmup_steps=0,
                                     num_training_steps=num_training_steps)
    else:
        lr_scheduler = None

    return Settings(model, optimizer, train_dataloader, val_dataloader,
                    lr_scheduler, args.num_epochs, num_training_steps, device)


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
        train_metrics = get_ner_metrics(model, settings.train_dataloader,
                                        settings.device)
        val_metrics = get_ner_metrics(model, settings.val_dataloader,
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
def main() -> None:
    """ Main function """

    args = get_args()
    out_dir = args.out_dir

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    model_name = args.model_name
    train_dataloader, val_dataloader = get_dataloaders(args)

    if args.seed:
        set_random_seed(45)
    settings = initialize_model(args, train_dataloader, val_dataloader)

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
