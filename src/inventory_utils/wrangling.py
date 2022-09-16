"""
Wrangling
~~~

Functions for wrangling, cleaning, and splitting data.

Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import io
import logging
import math
import re
import sys
from typing import List, Optional, TextIO, Tuple, Union

import numpy as np
import pandas as pd
import pytest
from numpy import array
from pandas.testing import assert_frame_equal
from sklearn.model_selection import train_test_split

from .aliases import TaggedBatch
from .constants import ID2NER_TAG
from .custom_classes import Splits


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

    df = pd.read_csv(file, dtype=str)

    if not all(map(lambda c: c in df.columns, ['id', 'title', 'abstract'])):
        sys.exit(f'Data file {file.name} must contain columns '
                 'labeled "title" and "abstract".')

    df.fillna('', inplace=True)
    df = df[~df.duplicated('id')]
    df = df[df['id'] != '']

    for col in ['title', 'abstract']:
        df[col] = df[col].apply(strip_xml)
        df[col] = df[col].apply(strip_newlines)

    df = concat_title_abstract(df)

    return df


# ---------------------------------------------------------------------------
def test_preprocess_data() -> None:
    """ Test preprocess_data() """

    in_fh = io.StringIO('id,title,abstract\n'
                        '123,A Descriptive Title,A <i>detailed</i> abstract.\n'
                        '456,Another title,Another abstract.\n'
                        '456,Another title,Another abstract.\n'
                        ',This one should go,now\n')

    out_df = pd.DataFrame(
        [[
            '123', 'A Descriptive Title', 'A detailed abstract.',
            'A Descriptive Title. A detailed abstract.'
        ],
         [
             '456', 'Another title', 'Another abstract.',
             'Another title. Another abstract.'
         ]],
        columns=['id', 'title', 'abstract', 'title_abstract'])

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
def join_commas(ls: List[str]) -> str:
    """
    Create a string by placing a comma and space between each element in a 
    list of strings.

    Parameters
    `ls`: List of strings

    Return: Joined string
    """

    return ', '.join(ls)


# ---------------------------------------------------------------------------
def test_join_commas() -> None:
    """ Test join_commas() """

    assert join_commas(['foo']) == 'foo'
    assert join_commas(['foo', 'bar', 'baz']) == 'foo, bar, baz'


# ---------------------------------------------------------------------------
def chunk_rows(
        in_item: Union[pd.DataFrame, pd.Series],
        chunk_size: Optional[int]) -> List[Union[pd.DataFrame, pd.Series]]:
    """
    Separate input dataframe or series into a list of dataframes (or series),
    each with ~`chunk_size` rows.

    Parameters:
    `in_item`: Input dataframe or series
    `chunk_size`: Maximum number of rows per chunk

    Return: List of dataframes or series
    """

    if not chunk_size:
        return [in_item]

    logging.debug('Splitting data into ~%d-row chunks', chunk_size)
    chunks = []
    num_chunks = math.ceil(len(in_item) / chunk_size)
    for chunk in np.array_split(in_item, num_chunks):
        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
def test_chunk_df() -> None:
    """ Test chunk_df() """

    in_df = pd.DataFrame([['foo', 'bar'], ['baz', 'qux'], ['quux', 'quuz'],
                          ['corge', 'grault']],
                         columns=['col_a', 'col_b'])

    # Return whole dataframe if no chunk size
    chunks = chunk_rows(in_df, None)
    assert len(chunks) == 1

    chunks = chunk_rows(in_df, 2)
    assert len(chunks) == 2

    chunks = chunk_rows(in_df, 1)
    assert len(chunks) == 4

    chunks = chunk_rows(in_df, 3)
    assert len(chunks) == 2

    # Also handles pd.Series
    chunks = chunk_rows(in_df['col_a'], 2)
    assert len(chunks) == 2


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    sys.exit('This file is a module, and is not meant to be run.')
