#!/usr/bin/env python3
"""
Purpose: Train NER model from pretrained BERT
Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import argparse
import copy
import os
import sys
from typing import Any, List, NamedTuple, Optional, Tuple, cast

import torch
from datasets import load_metric
from torch.functional import Tensor
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, get_scheduler

from ner_data_handler import RunParams, get_dataloader
from utils import (ARGS_MAP, ID2NER_TAG, NER_TAG2ID, CustomHelpFormatter,
                   Metrics, Settings, make_filenames, save_loss_plot,
                   save_model, set_random_seed)

# ---------------------------------------------------------------------------
""" Type Aliases """
LabeledBatch = List[List[int]]
TaggedBatch = List[List[str]]


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    train_file: str
    val_file: str
    out_dir: str
    model_name: str
    learning_rate: float
    weight_decay: float
    use_default_values: bool
    num_training: int
    num_epochs: int
    batch_size: int
    lr_scheduler: bool
    model_checkpoint: Optional[str]


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

    model_params.add_argument(
        '-m',
        '--model_name',
        metavar='',
        type=str,
        default='biomed_roberta',
        help='Name of model',
        choices=[
            'bert', 'biobert', 'bioelectra', 'bioelectra_pmc',
            'biomed_roberta', 'biomed_roberta_chemprot',
            'biomed_roberta_rct_500', 'bluebert', 'bluebert_mimic3',
            'electramed', 'pubmedbert', 'pubmedbert_pmc', 'sapbert',
            'sapbert_mean_token', 'scibert'
        ])
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
    model_params.add_argument('-def',
                              '--use-default-values',
                              metavar='',
                              type=bool,
                              default=True,
                              help='Use default values in ner_utils.py')

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

    args = parser.parse_args()

    args = Args(args.train_file, args.val_file, args.out_dir, args.model_name,
                args.learning_rate, args.weight_decay, args.use_default_values,
                args.num_training, args.num_epochs, args.batch_size,
                args.lr_scheduler, None)

    if args.use_default_values:
        args = get_default_args(args)

    return args


# ---------------------------------------------------------------------------
def get_default_args(args: Args) -> Args:
    """
    Get default options based on model

    `args`: Command-line arguments

    Return: Updated arguments
    """

    model_name = args.model_name
    model_checkpoint = ARGS_MAP[model_name][0]
    batch_size = ARGS_MAP[model_name][1]
    learning_rate = ARGS_MAP[model_name][2]
    weight_decay = ARGS_MAP[model_name][3]
    use_scheduler = ARGS_MAP[model_name][4]

    return Args(args.train_file, args.val_file, args.out_dir, model_name,
                learning_rate, weight_decay, True, args.num_training,
                args.num_epochs, batch_size, use_scheduler, model_checkpoint)


# ---------------------------------------------------------------------------
def get_dataloaders(args, model_name: str) -> Tuple[DataLoader, DataLoader]:
    """
    Generate training and validation dataloaders

    `args`: Command-line arguments

    Return: training dataloader, validation dataloader
    """

    print('Generating training and validation dataloaders ...')
    print('=' * 30)

    params = RunParams(model_name, args.batch_size, args.num_training)
    train_dataloader = get_dataloader(args.train_file, params)
    val_dataloader = get_dataloader(args.val_file, params)

    print('Finished generating dataloaders!')
    print('=' * 30)

    return train_dataloader, val_dataloader


# ---------------------------------------------------------------------------
def initialize_model(model_name: str, args: Args, train_dataloader: DataLoader,
                     val_dataloader: DataLoader) -> Settings:
    """
    Initialize the model and get settings

    `model_name`: Trained model name
    `args`: Command-line arguments
    `train_dataloader`: Training dataloader
    `val_dataloader`: Validation dataloader

    Return: training settings including model
    """

    print('Initializing', model_name, 'model ...')
    print('=' * 30)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, id2label=ID2NER_TAG, label2id=NER_TAG2ID)
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
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

    return Settings(model, optimizer, train_dataloader, val_dataloader,
                    lr_scheduler, args.num_epochs, num_training_steps, device)


# ---------------------------------------------------------------------------
def train(settings: Settings) -> Tuple:
    """
    Train the classifier

    Parameters:
    `settings`: Model settings (NamedTuple)
    """

    model = settings.model
    model.train()
    progress_bar = tqdm(range(settings.num_training_steps))
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

    return best_model, best_epoch, best_val.f1, train_losses, val_losses


# ---------------------------------------------------------------------------
def train_epoch(settings: Settings, progress_bar: tqdm) -> float:
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
    # calc_precision = load_metric('precision')
    # calc_recall = load_metric('recall')
    # calc_f1 = load_metric('f1')
    calc_seq_metrics = load_metric('seqeval')
    total_loss = 0.
    num_seen_datapoints = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        num_seen_datapoints += len(batch['input_ids'])
        predictions = torch.argmax(outputs.logits, dim=-1)  # Diff from class
        predictions = predictions.detach().cpu().clone().numpy()

        labels = cast(Tensor, batch['labels'])
        labels = labels.detach().cpu().clone().numpy()

        pred_labels, true_labels = convert_to_tags(predictions, labels)

        calc_seq_metrics.add_batch(predictions=pred_labels,
                                   references=true_labels)

        total_loss += outputs.loss.item()
    total_loss /= num_seen_datapoints

    precision, recall, f1 = extract_metrics(calc_seq_metrics.compute())

    return Metrics(precision, recall, f1, total_loss)


# ---------------------------------------------------------------------------
def extract_metrics(metric_dict: Optional[dict]) -> List[float]:
    """ Extract precision, recall, and F1 """

    if not metric_dict:
        sys.exit('Unable to calculate metrics.')

    return [
        metric_dict[f'overall_{metric}']
        for metric in ['precision', 'recall', 'f1']
    ]


# ---------------------------------------------------------------------------
def convert_to_tags(
        batch_predictions: LabeledBatch,
        batch_labels: LabeledBatch) -> Tuple[TaggedBatch, TaggedBatch]:
    """
    Convert numeric labels to string tags

    `batch_predictions`: Predicted numeric labels of batch of sequences
    `batch_labels`: True numeric labels of batch of sequences

    Return: Lists of tagged sequences of tokens from predictions and true labels
    """

    true_labels = [[
        ID2NER_TAG[token_label] for token_label in seq_labels
        if token_label != -100
    ] for seq_labels in batch_labels]
    pred_labels = [[
        ID2NER_TAG[token_pred]
        for (token_pred, token_label) in zip(seq_preds, seq_labels)
        if token_label != -100
    ] for seq_preds, seq_labels in zip(batch_predictions, batch_labels)]

    return pred_labels, true_labels


# ---------------------------------------------------------------------------
def test_convert_to_tags() -> None:
    """ Test convert_to_tags """

    # Inputs
    predictions = [[0, 0, 1, 2, 2, 0, 3, 4, 0], [0, 0, 0, 1, 0]]
    labels = [[-100, 0, 1, 2, 2, 0, 3, 4, -100], [-100, 0, 0, 3, -100]]

    # Expected outputs
    exp_pred = [['O', 'B-COM', 'I-COM', 'I-COM', 'O', 'B-FUL', 'I-FUL'],
                ['O', 'O', 'B-COM']]
    exp_labels = [['O', 'B-COM', 'I-COM', 'I-COM', 'O', 'B-FUL', 'I-FUL'],
                  ['O', 'O', 'B-FUL']]

    res_pred, res_labels = convert_to_tags(predictions, labels)

    assert exp_pred == res_pred
    assert exp_labels == res_labels


# ---------------------------------------------------------------------------
# class Trainer():
#     """
#      Handles training of the model
#     """
#     def __init__(self, model, optimizer, train_dataloader, val_dataloader,
#                  lr_scheduler, num_epochs, num_training_steps, device):
#         """
#         :param model: PyTorch model
#         :param optimizer: optimizer used
#         :param train_dataloader: DataLoader containing data used for training
#         :param val_dataloader: DataLoader containing data used for validation
#         :param lr_scheduler: learning rate scheduler; could be equal to None if no lr_scheduler is used
#         :param num_epochs: number of epochs to train the model for
#         :param num_training_steps: total number of training steps
#         :param device: device used for training; equal to 'cuda' if GPU is available
#         """
#         self.model = model
#         self.optimizer = optimizer
#         self.train_dataloader = train_dataloader
#         self.val_dataloader = val_dataloader
#         self.lr_scheduler = lr_scheduler
#         self.num_epochs = num_epochs
#         self.num_training_steps = num_training_steps
#         self.device = device

#     def evaluate(self, dataloader):
#         """
#         Computes and returns metrics (P, R, F1 score, loss) of a model on data present in a dataloader
#         :param dataloader: DataLoader containing tokenized text entries and corresponding labels
#         :return: precision, recall, F1 score, loss
#         """
#         metric = load_metric("seqeval")
#         total_loss = 0
#         num_seen_datapoints = 0

#         for batch in dataloader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             with torch.no_grad():
#                 outputs = self.model(**batch)
#             num_seen_datapoints += len(batch['input_ids'])
#             logits = outputs.logits
#             predictions = logits.argmax(dim=-1)
#             loss = outputs.loss

#             labels = batch["labels"]
#             pred_labels, true_labels = self.postprocess(predictions, labels)
#             metric.add_batch(predictions=pred_labels, references=true_labels)

#             total_loss += loss.item()
#         total_loss /= num_seen_datapoints
#         results = metric.compute()
#         p, r, f1, _ = self.get_metrics(results)
#         return p, r, f1, total_loss

#     def train_epoch(self, progress_bar):
#         """
#         Handles training of the model over one epoch
#         :param progress_bar: tqdm instance for tracking progress
#         """
#         train_loss = 0
#         num_train = 0
#         for batch in self.train_dataloader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             num_train += len(batch['input_ids'])
#             outputs = self.model(**batch)
#             loss = outputs.loss
#             loss.backward()
#             train_loss += loss.item()

#             self.optimizer.step()
#             if self.lr_scheduler:
#                 self.lr_scheduler.step()
#             self.optimizer.zero_grad()
#             progress_bar.update(1)

#         return train_loss / num_train

#     def train(self):
#         """
#         Handles training of the model over all epochs
#         """
#         progress_bar = tqdm(range(num_training_steps))
#         self.model.train()

#         best_model = self.model
#         best_val_f1_score = 0
#         best_epoch = -1
#         train_losses = []
#         val_losses = []
#         for epoch in range(self.num_epochs):
#             train_loss = 0
#             # Training
#             train_loss = self.train_epoch(progress_bar)
#             train_losses.append(train_loss)

#             # Evaluation
#             self.model.eval()
#             train_p, train_r, train_f1, _ = self.evaluate(
#                 self.train_dataloader)
#             val_p, val_r, val_f1, val_loss = self.evaluate(self.val_dataloader)

#             if val_f1 > best_val_f1_score:
#                 best_model = copy.deepcopy(self.model)
#                 best_val_f1_score = val_f1
#                 best_epoch = epoch

#             val_losses.append(val_loss)
#             print(
#                 "Epoch", (epoch + 1),
#                 ": Train Loss: %.5f Precision: %.3f Recall: %.3f F1: %.3f || Val Loss: %.5f Precision: %.3f Recall: %.3f F1: %.3f"
#                 % (train_loss, train_p, train_r, train_f1, val_loss, val_p,
#                    val_r, val_f1))
#         self.best_model = best_model
#         self.best_epoch = best_epoch
#         self.best_f1_score = best_val_f1_score
#         return best_model, best_epoch, best_val_f1_score, train_losses, val_losses

#     def get_metrics(self, results):
#         """
#         Return metrics (Precision, recall, f1, accuracy)
#         """
#         return [
#             results[f"overall_{key}"]
#             for key in ["precision", "recall", "f1", "accuracy"]
#         ]

#     def postprocess(self, predictions, labels):
#         """
#         Postprocess true and predicted arrays (as indices) to the corresponding labels (eg 'B-RES', 'I-RES')
#         :param predictions: array corresponding to predicted labels (as indices)
#         :param labels: array corresponding to true labels (as indices)
#         :return: predicted and true labels (as tags)
#         """
#         predictions = predictions.detach().cpu().clone().numpy()
#         labels = labels.detach().cpu().clone().numpy()
#         true_labels = [[ID2NER_TAG[l] for l in label if l != -100]
#                        for label in labels]
#         pred_labels = [[
#             ID2NER_TAG[p] for (p, l) in zip(prediction, label) if l != -100
#         ] for prediction, label in zip(predictions, labels)]
#         return pred_labels, true_labels

#     def save_best_model(self, checkpt_filename):
#         """
#         Saves a model checkpoint, epoch and F1 score to file
#         :param checkpt_filename: filename under which the model checkpoint will be saved
#         """
#         torch.save(
#             {
#                 'model_state_dict': self.best_model.state_dict(),
#                 'epoch': self.best_epoch,
#                 'f1_val': self.best_f1_score,
#             }, checkpt_filename)

#     def plot_losses(self, losses, labels, img_filename):
#         """
#         Plots training and val losses
#         :param num_epochs: total number of epochs the model was trained on; corresponds to length of the losses array
#         :param losses: array corresponding to [train_losses, val_losses]
#         :param img_filename: filename under which to save the image
#         :return: Generated plot
#         """
#         x = [i for i in range(self.num_epochs)]
#         df = pd.DataFrame({'Epoch': x})
#         for loss_arr, label in zip(losses, labels):
#             df[label] = loss_arr
#         fig = px.line(df, x="Epoch", y=labels, title='Train/Val Losses')
#         fig.show()
#         fig.write_image(img_filename)
#         return fig


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()
    out_dir = args.out_dir

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    model_name = ARGS_MAP[args.model_name][0]
    train_dataloader, val_dataloader = get_dataloaders(args, model_name)

    set_random_seed(45)
    settings = initialize_model(model_name, args, train_dataloader,
                                val_dataloader)

    print('Starting model training...')
    print('=' * 30)

    model, epoch, f1, train_losses, val_losses = train(settings)

    checkpt_filename, img_filename = make_filenames(out_dir, args.model_name)

    save_model(model, epoch, f1, checkpt_filename)
    save_loss_plot(train_losses, val_losses, img_filename)

    print('Done. Saved best checkpoint to', checkpt_filename)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
