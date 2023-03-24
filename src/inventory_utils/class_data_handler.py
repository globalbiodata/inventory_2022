"""
Classification Data Handler
~~~

Functions for creating a dataloader for the classification task,
which includes preprocessing and tokenization.

Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import random
import sys
from functools import partial
from typing import List, NamedTuple, Optional, TextIO, Tuple

import pandas as pd
from datasets import ClassLabel, Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer

from .wrangling import preprocess_data


# ---------------------------------------------------------------------------
class DataFields(NamedTuple):
    """
    Fields in data used for training and classification

    `predictive`: Column used for prediction
    `descriptive_labels`: Descriptions of the classification labels
    `labels`: Column containing labels (optional)
    """
    predictive: str
    descriptive_labels: List[str]
    labels: Optional[str] = None


# ---------------------------------------------------------------------------
class RunParams(NamedTuple):
    """
    Model and run parameters

    `model_name`: Huggingface model name
    `batch_size`: Tokenization batch size
    `max_len`: Tokenization max length
    `num_train`: Number of training datapoints (optional)
    """
    model_name: str
    batch_size: int
    max_len: int
    num_train: Optional[int] = None


# ---------------------------------------------------------------------------
def get_dataloader(file: TextIO, fields: DataFields,
                   run_params: RunParams) -> DataLoader:
    """
    Preprocess data and create dataloader

    Parameters:
    `file`: Input file handle
    `fields`: Fields in data used for training and classification
    `run_params`: Model and run parameters

    Return:
    A `DataLoader` with preprocessed data
    """

    df = preprocess_data(file)

    data_loader = generate_dataloader(df, file.name, fields, run_params)

    return data_loader


# ---------------------------------------------------------------------------
def generate_dataloader(df: pd.DataFrame, filename: str, fields: DataFields,
                        params: RunParams) -> DataLoader:
    """
    Generate dataloader from preprocessed data

    Parameters:
    `df`: Dataframe to be converted to `DataLoader`
    `filename`: Name of file from which `df` originates
    `fields`: Fields in data used for training and classification
    `params`: Model and run parameters

    Return:
    A `DataLoader` of preprocessed data
    """

    if fields.predictive not in df.columns:
        sys.exit(f'Predictive field column "{fields.predictive}" '
                 f'not in file {filename}.')

    if fields.labels and fields.labels not in df.columns:
        sys.exit(f'Labels field column "{fields.labels}" '
                 f'not in file {filename}.')

    text, labels = get_text_labels(df, fields)

    class_labels = ClassLabel(num_classes=2, names=fields.descriptive_labels)

    tokenizer = AutoTokenizer.from_pretrained(params.model_name)

    dataset = tokenize_text(text, labels, class_labels, tokenizer,
                            params.max_len)

    if params.num_train:
        dataset = dataset.select(
            random.sample(range(dataset.num_rows), k=params.num_train))

    return DataLoader(dataset, batch_size=params.batch_size)  # type:ignore


# ---------------------------------------------------------------------------
def get_text_labels(df: pd.DataFrame, fields: DataFields) -> Tuple[List, List]:
    """
    Get lists of predictive text and (optionally) labels

    Parameters:
    `df`: Dataframe containing `fields.predictive`
    `fields`: Specification of column names

    Return:
    A tuple of lists: predictive text, labels
    """

    text = df[fields.predictive].tolist()

    labels = []
    if fields.labels:
        labels = df[fields.labels].tolist()

    return text, labels


# ---------------------------------------------------------------------------
def test_get_text_labels() -> None:
    """ Test get_text_labels() """

    df = pd.DataFrame(
        [['Title 1', 'Abstract 1', 0], ['Title 2', 'Abstract 2', 1],
         ['Title 3', 'Abstract 3', 0]],
        columns=['title', 'abstract', 'score'])

    fields = DataFields('title', ['yes', 'no'])

    assert get_text_labels(df, fields) == (['Title 1', 'Title 2',
                                            'Title 3'], [])

    fields = DataFields('title', ['yes', 'no'], 'score')

    assert get_text_labels(df, fields) == (['Title 1', 'Title 2',
                                            'Title 3'], [0, 1, 0])


# ---------------------------------------------------------------------------
def tokenize_text(text: List, labels: List, class_labels: ClassLabel,
                  tokenizer: PreTrainedTokenizer, max_len: int) -> Dataset:
    """
    Tokenize predictive text

    Parameters:
    `text`: A list of predictive text
    `labels`: A list of labels of `text`
    `class_labels`: Descriptive labels of data in `text`
    `tokenizer`: Pretrained tokenizer
    `max_len`: Max length used in tokenization

    Return:
    A tokenized and possibly labeled `Dataset`
    """

    data = {'text': text}
    if labels:
        data['labels'] = labels
    dataset = Dataset.from_dict(data)

    # Partially apply arguments to the tokenizer so it is ready to tokenize
    tokenize = partial(tokenizer,
                       padding='max_length',
                       max_length=max_len,
                       truncation=True)

    tokenized_dataset = dataset.map(lambda x: tokenize(x['text']),
                                    batched=True)

    if labels:
        tokenized_dataset = tokenized_dataset.cast_column(
            'labels', class_labels)

    tokenized_dataset = tokenized_dataset.remove_columns(['text'])
    tokenized_dataset.set_format("torch")

    return tokenized_dataset


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    sys.exit('This file is a module, and is not meant to be run.')
