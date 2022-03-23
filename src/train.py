#!/usr/bin/env python3
"""
Purpose: Train BERT model for article classification
Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import argparse
import copy
import os
from typing import NamedTuple, TextIO, Tuple

import pandas as pd
import plotly.express as px
import torch
from datasets import load_metric
from torch._C import NoneType
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import (AdamW, AutoModelForSequenceClassification,
                          get_scheduler)

from data_handler import DataHandler
from utils import MODEL_TO_HUGGINGFACE_VERSION


# ---------------------------------------------------------------------------
class Settings(NamedTuple):
    """ Trainer settings """

    model: str
    optimizer: AdamW
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    lr_scheduler: NoneType
    num_epochs: int
    num_training_steps: int
    device: torch.device


# ---------------------------------------------------------------------------
class Metrics(NamedTuple):
    """ Performance metrics  """

    precision: float
    recall: float
    f1: float
    loss: float


# ---------------------------------------------------------------------------
class Trainer():
    """
    Handles training of the model
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, settings: Settings):
        """
        :param model: PyTorch model
        :param optimizer: optimizer used
        :param train_dataloader: DataLoader containing data used for training
        :param val_dataloader: DataLoader containing data used for validation
        :param lr_scheduler: learning rate scheduler; None if no lr_scheduler
        :param num_epochs: number of epochs to train the model for
        :param num_training_steps: total number of training steps
        :param device: device used for training; 'cuda' if GPU is available
        """
        self.model = settings.model
        self.optimizer = settings.optimizer
        self.train_dataloader = settings.train_dataloader
        self.val_dataloader = settings.val_dataloader
        self.lr_scheduler = settings.lr_scheduler
        self.num_epochs = settings.num_epochs
        self.num_training_steps = settings.num_training_steps
        self.device = settings.device
        self.best_model = settings.model
        self.best_epoch = 0.
        self.best_f1_score = 0.

    # -----------------------------------------------------------------------
    def get_metrics(self, dataloader) -> Metrics:
        """
        Computes and returns metrics (P, R, F1 score) of a model on
        data present in a dataloader

        :param model: model used to compute the metrics
        :param dataloader: DataLoader containing tokenized text entries and
        corresponding labels

        :return: precision, recall, F1 score
        """
        precision = load_metric("precision")
        recall = load_metric("recall")
        f1 = load_metric("f1")
        total_loss = 0
        num_seen_datapoints = 0
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            num_seen_datapoints += len(batch['input_ids'])
            logits = outputs.logits
            loss = outputs.loss
            predictions = torch.argmax(logits, dim=-1)
            precision.add_batch(predictions=predictions,
                                references=batch["labels"])
            recall.add_batch(predictions=predictions,
                             references=batch["labels"])
            f1.add_batch(predictions=predictions, references=batch["labels"])
            total_loss += loss.item()
        total_loss /= num_seen_datapoints

        return Metrics(precision.compute()['precision'],
                       recall.compute()['recall'],
                       f1.compute()['f1'], total_loss)

    # -----------------------------------------------------------------------
    def train_epoch(self, progress_bar):
        """
        Handles training of the model over one epoch

        :param model: PyTorch model
        :param optimizer: optimizer used
        :param lr_scheduler: learning rate scheduler; None if no lr_scheduler
        :param train_dataloader: DataLoader containing data used for training
        :param device: device used for training; 'cuda' if GPU is available
        :param progress_bar: tqdm instance for tracking progress
        """
        train_loss = 0
        num_train = 0
        for batch in self.train_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            num_train += len(batch['input_ids'])
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            progress_bar.update(1)
        return train_loss / num_train

    # -----------------------------------------------------------------------
    def train(self):
        """
        Handles training of the model over all epochs

        :param model: PyTorch model
        :param train_dataloader: DataLoader containing data used for training
        :param val_dataloader: DataLoader containing data used for validation
        :param optimizer: optimizer used
        :param lr_scheduler: learning rate scheduler; None if no lr_scheduler
        :param num_epochs: number of epochs to train the model for
        :param num_training_steps:
        :param checkpt_name: name under which the checkpoint will be saved
        :param device: device used for training; 'cuda' if GPU is available

        :return best_model: model checkpt that has the highest F1 score on
        the validation data
        :return best_epoch: epoch corresponding to best_model
        :return train_losses: list of training loss values over all epochs;
        helpful for plotting
        :return val_losses: list of validation loss values over all epochs;
        helpful for plotting

        """
        progress_bar = tqdm(range(self.num_training_steps))
        self.model.train()
        best_model = self.model
        train_losses = []
        val_losses = []
        best_val = Metrics(0, 0, 0, 0)

        for epoch in range(self.num_epochs):
            # training
            train_loss = self.train_epoch(progress_bar)
            # evaluation
            self.model.eval()
            train_metrics = self.get_metrics(self.train_dataloader)
            val_metrics = self.get_metrics(self.val_dataloader)

            if val_metrics.f1 > best_val.f1:
                best_val = val_metrics
                best_train = train_metrics
                best_model = copy.deepcopy(self.model)
                best_epoch = epoch

            train_losses.append(train_metrics.loss)
            val_losses.append(val_metrics.loss)

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
        self.best_model = best_model
        self.best_epoch = best_epoch
        self.best_f1_score = best_val.f1

        return best_model, best_epoch, train_losses, val_losses

    # -----------------------------------------------------------------------
    def save_best_model(self, checkpt_filename):
        """
        Saves a model checkpoint, epoch and F1 score to file

        :param model: model to save
        :param epoch: num_epoch corresponding to trained model
        :param f1_score: F1 score obtained by the model on validation data
        :param checkpt_filename: filename under which the model checkpoint
        will be saved
        """
        torch.save(
            {
                'model_state_dict': self.best_model.state_dict(),
                'epoch': self.best_epoch,
                'f1_val': self.best_f1_score,
            }, checkpt_filename)

    # -----------------------------------------------------------------------
    def plot_losses(self, losses, labels, img_filename):
        """
        Plots training and val losses

        :param num_epochs: total number of epochs the model was trained on;
        corresponds to length of the losses array
        :param losses: array corresponding to [train_losses, val_losses]
        :param labels: labels used for plotting;
        usually ['Train Loss', 'Val Loss']

        :return: Generated plot
        """
        x = list(range(self.num_epochs))
        df = pd.DataFrame({'Epoch': x})
        for loss_arr, label in zip(losses, labels):
            df[label] = loss_arr
        fig = px.line(df, x="Epoch", y=labels, title='Train/Val Losses')
        fig.show()
        fig.write_image(img_filename)
        return fig


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    train_file: TextIO
    val_file: TextIO
    test_file: TextIO
    output_dir: str
    predictive_field: str
    labels_field: str
    descriptive_labels: str
    model_name: str
    max_len: int
    learning_rate: float
    weight_decay: float
    sanity_check: bool
    num_training: int
    num_epochs: int
    batch_size: int
    lr_scheduler: bool


# ---------------------------------------------------------------------------
def get_args():
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Train BERT model for article classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    inputs = parser.add_argument_group('Inputs and Outputs')
    data_info = parser.add_argument_group('Information on Data')
    model_params = parser.add_argument_group('Model Parameters')
    runtime_params = parser.add_argument_group('Runtime Parameters')

    inputs.add_argument('-t',
                        '--train-file',
                        metavar='FILE',
                        type=argparse.FileType('rt'),
                        default='data/train.csv',
                        help='Location of training file')
    inputs.add_argument('-v',
                        '--val-file',
                        metavar='FILE',
                        type=argparse.FileType('rt'),
                        default='data/val.csv',
                        help='Location of validation file')
    inputs.add_argument('-s',
                        '--test-file',
                        metavar='FILE',
                        type=argparse.FileType('rt'),
                        default='data/test.csv',
                        help='Location of test file')
    inputs.add_argument('-o',
                        '--output-dir',
                        metavar='DIR',
                        type=str,
                        default='output_dir/',
                        help='Directory to output checkpt and plot losses')

    data_info.add_argument(
        '-pred',
        '--predictive-field',
        metavar='PRED',
        type=str,
        default='title',
        help='Field in the dataframes to use for prediction',
        choices=['title', 'abstract', 'title-abstract'])
    data_info.add_argument(
        '-labs',
        '--labels-field',
        metavar='LABS',
        type=str,
        default='curation_score',
        help='Field in the dataframes corresponding to the scores (0, 1)')
    data_info.add_argument(
        '-desc',
        '--descriptive-labels',
        metavar='LAB',
        type=str,
        nargs=2,
        default=['not-bio-resource', 'bio-resource'],
        help='Descriptive labels corresponding to the [0, 1] numeric scores')

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

    runtime_params.add_argument('-check',
                                '--sanity-check',
                                action='store_true',
                                help="""True for sanity-check.
        Runs training on a smaller subset of the entire training data.""")
    runtime_params.add_argument(
        '-nt',
        '--num-training',
        metavar='INT',
        type=int,
        default=-1,
        help="""Number of data points to run training on.
        If -1, training is ran an all the data. Useful for debugging.""")
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

    runtime_params.add_argument(
        '-lr',
        '--lr-scheduler',
        action='store_true',
        help="True if using a Learning Rate Scheduler.")

    args = parser.parse_args()

    return Args(args.train_file, args.val_file, args.test_file,
                args.output_dir, args.predictive_field, args.labels_field,
                args.descriptive_labels, args.model_name, args.max_len,
                args.learning_rate, args.weight_decay, args.sanity_check,
                args.num_training, args.num_epochs, args.batch_size,
                args.lr_scheduler)


# ---------------------------------------------------------------------------
def get_dataloaders(args: Args) -> Tuple[str, DataLoader, DataLoader]:
    """ Generate the dataloaders """

    print('Generating train, val, test dataloaders ...')
    print('=' * 30)
    model_name = MODEL_TO_HUGGINGFACE_VERSION[args.model_name]
    data_handler = DataHandler(model_name, args.train_file.name,
                               args.val_file.name, args.test_file.name)
    data_handler.parse_abstracts_xml()
    data_handler.concatenate_title_abstracts()
    data_handler.generate_dataloaders(args.predictive_field, args.labels_field,
                                      args.descriptive_labels, args.batch_size,
                                      args.max_len, args.sanity_check,
                                      args.num_training)
    train_dataloader = data_handler.train_dataloader
    val_dataloader = data_handler.val_dataloader
    print('Finished generating dataloaders!')
    print('=' * 30)

    return model_name, train_dataloader, val_dataloader


# ---------------------------------------------------------------------------
def initialize_model(model_name: str, args: Args, train_dataloader: DataLoader,
                     val_dataloader: DataLoader) -> Settings:
    """ Initialize the model and get settings  """

    print('Initializing', model_name, 'model ...')
    print('=' * 30)
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
                    lr_scheduler, args.num_epochs, num_training_steps, device)


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    model_name, train_dataloader, val_dataloader = get_dataloaders(args)

    settings = initialize_model(model_name, args, train_dataloader,
                                val_dataloader)

    # Model Training
    print('Starting model training...')
    print('=' * 30)

    trainer = Trainer(settings)
    _, best_epoch, train_losses, val_losses = trainer.train()

    # Save best checkpoint
    checkpt_filename = args.output_dir + 'checkpt_' + args.model_name + '_' + \
        str(best_epoch + 1) + '_epochs'
    trainer.save_best_model(checkpt_filename)
    print('Saved best checkpt to', checkpt_filename)

    # Plot losses
    img_filename = args.output_dir + args.model_name + '_' + str(
        best_epoch + 1) + '_epochs.png'
    trainer.plot_losses([train_losses, val_losses], ['Train Loss', 'Val Loss'],
                        img_filename)
    print('=' * 30)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()