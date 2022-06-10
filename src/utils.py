"""
Purpose: Provide shared data structures and classes
Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import argparse
import io
import os
import re
import sys
from typing import Any, List, NamedTuple, TextIO, Tuple

import pandas as pd
import plotly.express as px
import pytest
import torch
from pandas.testing import assert_frame_equal
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW

# ---------------------------------------------------------------------------
# Mapping from NER tag to ID
NER_TAG2ID = {'O': 0, 'B-COM': 1, 'I-COM': 2, 'B-FUL': 3, 'I-FUL': 4}

# Mapping from ID to NER tag
ID2NER_TAG = {v: k for k, v in NER_TAG2ID.items()}


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
def set_random_seed(seed):
    """
    Sets random seed for deterministic outcome of ML-trained models
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
def get_torch_device() -> torch.device:
    """
    Get device for torch

    Returns:
    `torch.device` either "cuda" or "cpu"
    """

    return torch.device('cuda') if torch.cuda.is_available() else torch.device(
        'cpu')


# ---------------------------------------------------------------------------
def split_df(df: pd.DataFrame, rand_seed: bool, splits: List[float]) -> Splits:
    """
    Split manually curated data into train, validation and test sets

    `df`: Manually curated classification data
    `rand_seed`: Optionally use random seed
    `splits`: Proportions of data for [train, validation, test]

    Return:
    train, validation, test dataframes
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

    Returns:
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
def concat_title_abstract(df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate abstract and title columns

    Parameters:
    `df`: Dataframe with columns "title" and "abstract"

    Returns:
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

    Parameter:
    `text`: String that may be missing final puncturation

    Returns:
    `text` With final punctuation
    """

    return text if text[-1] in '.?!' else text + '.'


# ---------------------------------------------------------------------------
def test_add_period() -> None:
    """ Test add_poeriod() """

    assert add_period('A statement.') == 'A statement.'
    assert add_period('A question?') == 'A question?'
    assert add_period('An exclamation!') == 'An exclamation!'
    assert add_period('An incomplete') == 'An incomplete.'


# ---------------------------------------------------------------------------
def preprocess_data(file: TextIO) -> pd.DataFrame:
    """
    Strip XML tags and concatenate title and abstract columns

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
def make_filenames(out_dir: str) -> Tuple[str, str]:
    """
    Make output filename

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
def save_model(model: Any, model_name: str, filename: str) -> None:
    """
    Save model checkpoint, epoch, and F1 score to file

    Parameters:
    `model`: Model to save
    `filename`: Name of file for saving model
    """

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'model_name': model_name
        }, filename)


# ---------------------------------------------------------------------------
def save_train_stats(df: pd.DataFrame, filename: str) -> None:
    """
    Save training performance metrics to file
    """

    df.to_csv(filename, index=False)


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
        'Epoch': list(range(1,
                            len(val_losses) + 1)),
        'Train': train_losses,
        'Validation': val_losses
    })

    fig = px.line(df,
                  x='Epoch',
                  y=['Train', 'Validation'],
                  title='Train and Validation Losses')

    fig.write_image(filename)
