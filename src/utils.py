"""
Purpose: Provide shared data structures and classes
Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import argparse
import io
import os
import re
import sys
from typing import (Any, BinaryIO, List, NamedTuple, Optional, TextIO, Tuple,
                    cast)

import pandas as pd
import pytest
import torch
from datasets import load_metric
from numpy import array
from pandas.testing import assert_frame_equal
from sklearn.model_selection import train_test_split
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW
from transformers import AutoModelForSequenceClassification as classifier
from transformers import AutoModelForTokenClassification as ner_classifier
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

# ---------------------------------------------------------------------------
NER_TAG2ID = {'O': 0, 'B-COM': 1, 'I-COM': 2, 'B-FUL': 3, 'I-FUL': 4}
"""
Mapping of NER tags to numerical labels

`key`: String NER label ("O", "B-COM", "I-COM", "B-FUL", "I-FUL")
`value`: Numerical label (1, 2, 3, 4)
"""

ID2NER_TAG = {v: k for k, v in NER_TAG2ID.items()}
"""
Mapping of numerical labels to NER tags

`key`: Numerical label (1, 2, 3, 4)
`value`: String NER label ("O", "B-COM", "I-COM", "B-FUL", "I-FUL")
"""


# ---------------------------------------------------------------------------
class CustomHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """ Custom Argparse help formatter """
    def _get_help_string(self, action):
        """ Suppress defaults that are None """
        if action.default is None:
            return action.help
        return super()._get_help_string(action)

    def _format_action_invocation(self, action):
        """ Show metavar only once """
        if not action.option_strings:
            metavar, = self._metavar_formatter(action, action.dest)(1)
            return metavar
        parts = []
        if action.nargs == 0:
            parts.extend(action.option_strings)
        else:
            default = action.dest.upper()
            args_string = self._format_args(action, default)
            for option_string in action.option_strings:
                parts.append('%s' % option_string)
            parts[-1] += ' %s' % args_string
        return ', '.join(parts)


# ---------------------------------------------------------------------------
class Splits(NamedTuple):
    """
    Training, validation, and test dataframes

    `train`: Training data
    `val`: Validation data
    `test`: Test data
    """
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


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
    """

    model: Any
    optimizer: AdamW
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    lr_scheduler: Any
    num_epochs: int
    num_training_steps: int
    device: torch.device


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
# Type Aliases
TaggedBatch = List[List[str]]


# ---------------------------------------------------------------------------
def set_random_seed(seed: int):
    """
    Set random seed for deterministic outcome of ML-trained models

    `seed`: Value to use for seed
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
def get_torch_device() -> torch.device:
    """
    Get device for torch

    Return:
    `torch.device` either "cuda" or "cpu"
    """

    return torch.device('cuda') if torch.cuda.is_available() else torch.device(
        'cpu')


# ---------------------------------------------------------------------------
def get_classif_model(checkpoint_fh: BinaryIO,
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
def get_ner_model(
        checkpoint_fh: BinaryIO,
        device: torch.device) -> Tuple[Any, str, PreTrainedTokenizer]:
    """
    Instatiate predictive NER model from checkpoint

    Params:
    `checkpoint_fh`: Model checkpoint filehandle
    `device`: The `torch.device` to use

    Return:
    Model instance from checkpoint, model name, and tokenizer
    """

    checkpoint = torch.load(checkpoint_fh, map_location=device)
    model_name = checkpoint['model_name']
    model = ner_classifier.from_pretrained(model_name,
                                           id2label=ID2NER_TAG,
                                           label2id=NER_TAG2ID)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, model_name, tokenizer


# ---------------------------------------------------------------------------
def split_df(df: pd.DataFrame, rand_seed: bool, splits: List[float]) -> Splits:
    """
    Split manually curated data into train, validation and test sets

    Parameters:
    `df`: Manually curated classification data
    `rand_seed`: Optionally use random seed
    `splits`: Proportions of data for [train, validation, test]

    Return:
    `Splits` containing train, validation, and test dataframes
    """

    seed = 241 if rand_seed else None

    _, val_split, test_split = splits
    val_test_split = val_split + test_split

    train, val_test = train_test_split(df,
                                       test_size=val_test_split,
                                       random_state=seed)
    val, test = train_test_split(val_test,
                                 test_size=test_split / val_test_split,
                                 random_state=seed)

    return Splits(train, val, test)


# ---------------------------------------------------------------------------
@pytest.fixture(name='unsplit_data')
def fixture_unsplit_data() -> pd.DataFrame:
    """ Example dataframe for testing splitting function """

    df = pd.DataFrame([[123, 'First title', 'First abstract', 0],
                       [456, 'Second title', 'Second abstract', 1],
                       [789, 'Third title', 'Third abstract', 0],
                       [321, 'Fourth title', 'Fourth abstract', 1],
                       [654, 'Fifth title', 'Fifth abstract', 0],
                       [987, 'Sixth title', 'Sixth abstract', 1],
                       [741, 'Seventh title', 'Seventh abstract', 0],
                       [852, 'Eighth title', 'Eighth abstract', 1]],
                      columns=['id', 'title', 'abstract', 'curation_score'])

    return df


# ---------------------------------------------------------------------------
def test_random_split(unsplit_data: pd.DataFrame) -> None:
    """ Test that split_df() gives correct proportions """

    in_df = unsplit_data

    train, val, test = split_df(in_df, False, [0.5, 0.25, 0.25])

    assert len(train.index) == 4
    assert len(val.index) == 2
    assert len(test.index) == 2


# ---------------------------------------------------------------------------
def test_seeded_split(unsplit_data: pd.DataFrame) -> None:
    """ Test that split_df() behaves deterministically """

    in_df = unsplit_data

    train, val, test = split_df(in_df, True, [0.5, 0.25, 0.25])

    assert list(train['id'].values) == [321, 789, 741, 654]
    assert list(val['id'].values) == [987, 456]
    assert list(test['id'].values) == [852, 123]


# ---------------------------------------------------------------------------
def strip_xml(text: str) -> str:
    """
    Strip XML tags from a string

    Parameters:
    `text`: String possibly containing XML tags

    Return:
    String without XML tags
    """
    # If header tag between two adjacent strings, replace with a space
    pattern = re.compile(
        r'''(?<=[\w.?!]) # Header tag must be preceded by word
            (</?h[\d]>) # Header tag has letter h and number
            (?=[\w]) # Header tag must be followed by word''', re.X)
    text = re.sub(pattern, ' ', text)

    # Remove all other XML tags
    text = re.sub(r'<[\w/]+>', '', text)

    return text


# ---------------------------------------------------------------------------
def test_strip_xml() -> None:
    """ Test strip_xml() """

    assert strip_xml('<h4>Supplementary info</h4>') == 'Supplementary info'
    assert strip_xml('H<sub>2</sub>O<sub>2</sub>') == 'H2O2'
    assert strip_xml(
        'the <i>Bacillus pumilus</i> group.') == 'the Bacillus pumilus group.'

    # If there are not spaces around header tags, add them
    assert strip_xml(
        'MS/MS spectra.<h4>Availability') == 'MS/MS spectra. Availability'
    assert strip_xml('http://proteomics.ucsd.edu/Software.html<h4>Contact'
                     ) == 'http://proteomics.ucsd.edu/Software.html Contact'
    assert strip_xml(
        '<h4>Summary</h4>Neuropeptides') == 'Summary Neuropeptides'
    assert strip_xml('<h4>Wow!</h4>Go on') == 'Wow! Go on'


# ---------------------------------------------------------------------------
def strip_newlines(text: str) -> str:
    """
    Remove all newline characters from string

    Parameters:
    `text`: String

    Return: string without newlines
    """

    return re.sub('\n', '', text)


# ---------------------------------------------------------------------------
def test_strip_newlines() -> None:
    """ Test strip_newlines() """

    assert strip_newlines('Hello, \nworld!') == 'Hello, world!'


# ---------------------------------------------------------------------------
def concat_title_abstract(df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate abstract and title columns

    Parameters:
    `df`: Dataframe with columns "title" and "abstract"

    Return:
    A `pd.DataFrame` with new column "title_abstract"
    """

    df['title_abstract'] = df['title'].map(add_period) + ' ' + df['abstract']

    return df


# ---------------------------------------------------------------------------
def test_concat_title_abstract() -> None:
    """ Test concat_title_abstract() """

    in_df = pd.DataFrame([['A Descriptive Title', 'A detailed abstract.']],
                         columns=['title', 'abstract'])

    out_df = pd.DataFrame([[
        'A Descriptive Title', 'A detailed abstract.',
        'A Descriptive Title. A detailed abstract.'
    ]],
                          columns=['title', 'abstract', 'title_abstract'])

    assert_frame_equal(concat_title_abstract(in_df), out_df)


# ---------------------------------------------------------------------------
def add_period(text: str) -> str:
    """
    Add period to end of sentence if punctuation not present

    Parameters:
    `text`: String that may be missing final puncturation

    Return:
    `text` with final punctuation
    """

    if not text:
        return ''

    return text if text[-1] in '.?!' else text + '.'


# ---------------------------------------------------------------------------
def test_add_period() -> None:
    """ Test add_poeriod() """

    assert add_period('') == ''
    assert add_period('A statement.') == 'A statement.'
    assert add_period('A question?') == 'A question?'
    assert add_period('An exclamation!') == 'An exclamation!'
    assert add_period('An incomplete') == 'An incomplete.'


# ---------------------------------------------------------------------------
def preprocess_data(file: TextIO) -> pd.DataFrame:
    """
    Strip XML tags and newlines and concatenate title and abstract columns

    Parameters:
    `file`: Input file handle

    Returns:
    a `pd.DataFrame` of preprocessed data
    """

    df = pd.read_csv(file)

    if not all(map(lambda c: c in df.columns, ['title', 'abstract'])):
        sys.exit(f'Data file {file.name} must contain columns '
                 'labeled "title" and "abstract".')

    df.fillna('', inplace=True)

    for col in ['title', 'abstract']:
        df[col] = df[col].apply(strip_xml)
        df[col] = df[col].apply(strip_newlines)

    df = concat_title_abstract(df)

    return df


# ---------------------------------------------------------------------------
def test_preprocess_data() -> None:
    """ Test preprocess_data() """

    in_fh = io.StringIO('title,abstract\n'
                        'A Descriptive Title,A <i>detailed</i> abstract.\n'
                        'Another title,Another abstract.')

    out_df = pd.DataFrame([[
        'A Descriptive Title', 'A detailed abstract.',
        'A Descriptive Title. A detailed abstract.'
    ],
                           [
                               'Another title', 'Another abstract.',
                               'Another title. Another abstract.'
                           ]],
                          columns=['title', 'abstract', 'title_abstract'])

    assert_frame_equal(preprocess_data(in_fh), out_df)


# ---------------------------------------------------------------------------
def convert_to_tags(batch_predictions: array,
                    batch_labels: array) -> Tuple[TaggedBatch, TaggedBatch]:
    """
    Convert numeric labels to string tags

    Parameters:
    `batch_predictions`: Predicted numeric labels of batch of sequences
    `batch_labels`: True numeric labels of batch of sequences

    Return: Lists of tagged sequences of tokens
    from predictions and true labels
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
    predictions = array([[0, 0, 1, 2, 2, 0, 3, 4, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0]])
    labels = array([[-100, 0, 1, 2, 2, 0, 3, 4, -100],
                    [-100, 0, 0, 3, -100, -100, -100, -100, -100]])

    # Expected outputs
    exp_pred = [['O', 'B-COM', 'I-COM', 'I-COM', 'O', 'B-FUL', 'I-FUL'],
                ['O', 'O', 'B-COM']]
    exp_labels = [['O', 'B-COM', 'I-COM', 'I-COM', 'O', 'B-FUL', 'I-FUL'],
                  ['O', 'O', 'B-FUL']]

    res_pred, res_labels = convert_to_tags(predictions, labels)

    assert exp_pred == res_pred
    assert exp_labels == res_labels


# ---------------------------------------------------------------------------
def get_classif_metrics(model: Any, dataloader: DataLoader,
                        device: torch.device) -> Metrics:
    """
    Compute classifier model performance metrics

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
def get_ner_metrics(model: Any, dataloader: DataLoader,
                    device: torch.device) -> Metrics:
    """
    Compute model performance metrics for NER model

    Parameters:
    `model`: Classification model
    `dataloader`: DataLoader containing tokenized text entries and
    corresponding labels
    `device`: Torch device

    Return:
    A `Metrics` NamedTuple
    """
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
    """
    Extract precision, recall, and F1

    Parameters:
    `metric_dict`: Dictionary of metrics

    Return: List of precision, recall, and F1
    """

    if not metric_dict:
        sys.exit('Unable to calculate metrics.')

    return [
        metric_dict[f'overall_{metric}']
        for metric in ['precision', 'recall', 'f1']
    ]


# ---------------------------------------------------------------------------
def make_filenames(out_dir: str) -> Tuple[str, str]:
    """
    Make output filename

    Parameters:
    `out_dir`: Output directory to be included in filename

    Return: Tuple['{out_dir}/checkpt.pt', '{out_dir}/train_stats.csv']
    """

    return os.path.join(out_dir,
                        'checkpt.pt'), os.path.join(out_dir, 'train_stats.csv')


# ---------------------------------------------------------------------------
def test_make_filenames() -> None:
    """ Test make_filenames """

    assert make_filenames('out/scibert') == ('out/scibert/checkpt.pt',
                                             'out/scibert/train_stats.csv')


# ---------------------------------------------------------------------------
def save_model(model: Any, model_name: str, train_metrics: Metrics,
               val_metrics: Metrics, filename: str) -> None:
    """
    Save model checkpoint, epoch, and F1 score to file

    Parameters:
    `model`: Model to save
    `model_name`: Model HuggingFace name
    `train_metrics`: Metrics on training set of best epoch
    `val_metrics`: Metrics on validation set of best epoch
    `filename`: Name of file for saving model
    """

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }, filename)


# ---------------------------------------------------------------------------
def save_train_stats(df: pd.DataFrame, filename: str) -> None:
    """
    Save training performance metrics to file

    Parameters:
    `df`: Training stats dataframe
    `filename`: Name of file for saving dataframe
    """

    df.to_csv(filename, index=False)


# ---------------------------------------------------------------------------
def save_metrics(metrics: Metrics, filename: str) -> None:
    """
    Save test metrics to csv file

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
if __name__ == '__main__':
    sys.exit('This file is a module, and is not meant to be run.')