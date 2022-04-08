#!/usr/bin/env python3
"""
Purpose: Train BERT model for article classification
Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import argparse
import copy
import logging
import os
from typing import Any, List, NamedTuple, Optional, TextIO, Tuple

import pandas as pd
import plotly.express as px
import torch
from datasets import load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import (AdamW, AutoModelForSequenceClassification,
                          get_scheduler)

from data_handler import get_dataloader, DataFields, RunParams
from utils import MODEL_TO_HUGGINGFACE_VERSION, CustomHelpFormatter


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    quiet: bool
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


# ---------------------------------------------------------------------------
class Settings(NamedTuple):
    """
    Settings used for model training

    `model`: Pretrained model
    `optimizer`: Training optimizer
    `train_dataloader`: `DataLoader` of training data
    `val_dataloader`: `DataLoader` of validation data
    `lr_scheduler`: Learning rate schedule (optional)
    `num_epochs`: Maximum number of training epochs
    `num_training_steps`: Maximum number of training steps
    (`num_epochs` * `num_training`)
    `device`: Torch device
    `quiet`: Run with minimal verbosity
    """

    model: Any
    optimizer: AdamW
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    lr_scheduler: Any
    num_epochs: int
    num_training_steps: int
    device: torch.device
    quiet: bool


# ---------------------------------------------------------------------------
class Metrics(NamedTuple):
    """
    Performance metrics

    `precision`: Model precision
    `recall`: Model recall
    `f1`: Model F1 score
    `loss`: Model loss
    """

    precision: float
    recall: float
    f1: float
    loss: float


# ---------------------------------------------------------------------------
def get_args():
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Train BERT model for article classification',
        formatter_class=CustomHelpFormatter)

    parser.add_argument('-q',
                        '--quiet',
                        help='Run with minimal verbosity',
                        action='store_true')

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

    model_params.add_argument(
        '-m',
        '--model-name',
        metavar='MODEL',
        type=str,
        default='scibert',
        help='Name of model',
        choices=[
            'bert', 'biobert', 'scibert', 'pubmedbert', 'pubmedbert_pmc',
            'bluebert', 'bluebert_mimic3', 'sapbert', 'sapbert_mean_token',
            'bioelectra', 'bioelectra_pmc', 'electramed', 'biomed_roberta',
            'biomed_roberta_chemprot', 'biomed_roberta_rct_500'
        ])
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

    args = parser.parse_args()

    return Args(args.quiet, args.train_file, args.val_file, args.out_dir,
                args.predictive_field, args.labels_field,
                args.descriptive_labels, args.model_name, args.max_len,
                args.learning_rate, args.weight_decay, args.num_training,
                args.num_epochs, args.batch_size, args.lr_scheduler)


# ---------------------------------------------------------------------------
def train(settings: Settings) -> Tuple:
    """
    Train the classifier

    Parameters:
    `settings`: Model settings (NamedTuple)
    """

    model = settings.model
    model.train()
    progress_bar = tqdm(range(
        settings.num_training_steps)) if not settings.quiet else None
    best_model = model
    train_losses = []
    val_losses = []
    best_val = Metrics(0, 0, 0, 0)
    best_train = Metrics(0, 0, 0, 0)
    best_epoch = 0

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
            best_epoch = epoch

        # Stop training once validation F1 goes down
        # Overfitting has begun
        if val_metrics.f1 < best_val.f1 and epoch > 0:
            break

        train_losses.append(train_metrics.loss)
        val_losses.append(val_metrics.loss)

        logging.info(f'Epoch {epoch + 1}:\n'
                     f'Train Loss: {train_loss:.5f}\n'
                     f'Val Loss: {val_metrics.loss:.5f}\n'
                     f'Train Precision: {train_metrics.precision:.3f}\n'
                     f'Train Recall: {train_metrics.recall:.3f}\n'
                     f'Train F1: {train_metrics.f1:.3f}\n'
                     f'Val Precision: {val_metrics.precision:.3f}\n'
                     f'Val Recall: {val_metrics.recall:.3f}\n'
                     f'Val F1: {val_metrics.f1:.3f}')

    logging.info('Finished model training!')
    logging.info('=' * 30)
    logging.info(f'Best Train Precision: {best_train.precision:.3f}\n'
                 f'Best Train Recall: {best_train.recall:.3f}\n'
                 f'Best Train F1: {best_train.f1:.3f}\n'
                 f'Best Val Precision: {best_val.precision:.3f}\n'
                 f'Best Val Recall: {best_val.recall:.3f}\n'
                 f'Best Val F1: {best_val.f1:.3f}\n')

    return best_model, best_epoch, best_val.f1, train_losses, val_losses


# ---------------------------------------------------------------------------
def train_epoch(settings: Settings, progress_bar: Optional[tqdm]) -> float:
    """
    Perform one epoch of model training

    Parameters:
    `settings`: Model settings (NamedTuple)
    `progress_bar`: tqdm instance for tracking progress

    Return: Train loss per observation
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
        if progress_bar:
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

    Returns:
    A `Metrics` NamedTuple
    """
    precision = load_metric("precision")
    recall = load_metric("recall")
    f1 = load_metric("f1")
    total_loss = 0.
    num_seen_datapoints = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        num_seen_datapoints += len(batch['input_ids'])
        logits = outputs.logits
        loss = outputs.loss
        predictions = torch.argmax(logits, dim=-1)
        precision.add_batch(predictions=predictions,
                            references=batch["labels"])
        recall.add_batch(predictions=predictions, references=batch["labels"])
        f1.add_batch(predictions=predictions, references=batch["labels"])
        total_loss += loss.item()
    total_loss /= num_seen_datapoints

    return Metrics(precision.compute()['precision'],
                   recall.compute()['recall'],
                   f1.compute()['f1'], total_loss)


# ---------------------------------------------------------------------------
def save_model(model: Any, epoch: int, f1: float, filename: str) -> None:
    """
    Save model checkpoint, epoch, and F1 score to file

    Parameters:
    `model`: Model to save
    `epoch`: Epochs used to train model
    `f1`: F1 score obtained by model
    `filename`: Name of file for saving model
    """

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'f1_val': f1
        }, filename)


# ---------------------------------------------------------------------------
def save_loss_plot(train_losses: List[float], val_losses: List[float],
                   filename: str) -> None:
    """
    Plot training and validation losses, and save to file

    Parameters:
    `train_losses`: Training losses
    `val_losses`: Validation losses
    `filename`: Name of file for saving plot
    """
    df = pd.DataFrame({
        'Epoch': list(range(1,len(val_losses) + 1)),
        'Train': train_losses,
        'Validation': val_losses
    })

    fig = px.line(df,
                  x='Epoch',
                  y=['Train', 'Validation'],
                  title='Train and Validation Losses')

    fig.write_image(filename)


# ---------------------------------------------------------------------------
def get_dataloaders(args: Args,
                    model_name: str) -> Tuple[DataLoader, DataLoader]:
    """ Generate the dataloaders """

    logging.info('Generating dataloaders ...')
    logging.info('=' * 30)

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

    logging.info('Finished generating dataloaders!')
    logging.info('=' * 30)

    return train_dataloader, val_dataloader


# ---------------------------------------------------------------------------
def initialize_model(model_name: str, args: Args, train_dataloader: DataLoader,
                     val_dataloader: DataLoader) -> Settings:
    """ Initialize the model and get settings  """

    logging.info(f'Initializing {model_name} model ...')
    logging.info('=' * 30)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               num_labels=2)
    optimizer = AdamW(model.parameters(),
                      lr=args.learning_rate,
                      weight_decay=args.weight_decay)
    num_training_steps = args.num_epochs * len(train_dataloader)
    if args.lr_scheduler:
        lr_scheduler = get_scheduler("linear",
                                     optimizer=optimizer,
                                     num_warmup_steps=0,
                                     num_training_steps=num_training_steps)
    else:
        lr_scheduler = None
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    return Settings(model, optimizer, train_dataloader, val_dataloader,
                    lr_scheduler, args.num_epochs, num_training_steps, device,
                    args.quiet)


# ---------------------------------------------------------------------------
def make_filenames(out_dir: str, model_name: str) -> Tuple[str, str]:
    """ Make output filenames """

    partial_name = os.path.join(out_dir, model_name + '_')

    return partial_name + 'checkpt.pt', partial_name + 'losses.png'


# ---------------------------------------------------------------------------
def test_make_filenames() -> None:
    """ Test make_filenames """

    assert make_filenames('out', 'scibert') == ('out/scibert_checkpt.pt',
                                                'out/scibert_losses.png')


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()
    out_dir = args.out_dir

    logging.basicConfig(level=logging.CRITICAL if args.quiet else logging.INFO,
                        format='%(message)s')

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    model_name = MODEL_TO_HUGGINGFACE_VERSION[args.model_name]

    train_dataloader, val_dataloader = get_dataloaders(args, model_name)

    settings = initialize_model(model_name, args, train_dataloader,
                                val_dataloader)

    logging.info('Starting model training...')
    logging.info('=' * 30)

    model, epoch, f1, train_losses, val_losses = train(settings)

    checkpt_filename, img_filename = make_filenames(out_dir, args.model_name)
    save_model(model, epoch, f1, checkpt_filename)
    save_loss_plot(train_losses, val_losses, img_filename)

    print('Done. Saved best checkpoint to', checkpt_filename)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
