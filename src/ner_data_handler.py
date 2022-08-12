"""
Purpose: Preprocess and tokenize data, create DataLoader
Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import random
import sys
from functools import partial
from typing import List, NamedTuple, Optional, cast

from datasets import load_dataset
from datasets.arrow_dataset import Batch
from datasets.dataset_dict import DatasetDict
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, DataCollatorForTokenClassification,
                          PreTrainedTokenizer)
from transformers.tokenization_utils_base import BatchEncoding

from utils import NER_TAG2ID


# ---------------------------------------------------------------------------
class RunParams(NamedTuple):
    """
    Model and run parameters

    `model_name`: Huggingface model name
    `batch_size`: Tokenization batch size
    `num_train`: Number of training datapoints (optional)
    """
    model_name: str
    batch_size: int
    num_train: Optional[int] = None


# ---------------------------------------------------------------------------
def get_dataloader(file: str, run_params: RunParams) -> DataLoader:
    """
    Preprocess data and create dataloader

    Parameters:
    `file`: Input file name
    `run_params`: Model and run parameters

    Return:
    A `DataLoader` with preprocessed data
    """

    dataset = load_dataset('pandas', data_files={'set': file})
    dataset = cast(DatasetDict, dataset)  # Cast for type checker

    tokenizer = AutoTokenizer.from_pretrained(run_params.model_name,
                                              add_prefix_space=True)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    tokenize_align_labels_with_tokenizer = partial(tokenize_align_labels,
                                                   tokenizer=tokenizer)
    tokenized_dataset = dataset.map(tokenize_align_labels_with_tokenizer,
                                    batched=True,
                                    remove_columns=dataset['set'].column_names)

    if run_params.num_train:
        tokenized_dataset['set'] = tokenized_dataset['set'].select(
            random.sample(range(dataset['set'].num_rows),
                          k=run_params.num_train))

    dataloader = DataLoader(
        tokenized_dataset['set'],  # type:ignore
        shuffle=True,
        collate_fn=collator,
        batch_size=run_params.batch_size)

    return dataloader


# ---------------------------------------------------------------------------
def tokenize_align_labels(dataset: Batch,
                          tokenizer: PreTrainedTokenizer) -> BatchEncoding:
    """
    Tokenize sequences of `words` and align numeric tags to tokens
    based on provided `ner_tags`

    Parameters:
    `dataset`: Batch of a `Dataset`
    `tokenizer`: Tokenizer for sequence tokenization

    Return: Batch of tokenized dataset with labels
    """

    tokenized_inputs = tokenizer(dataset['words'],
                                 truncation=True,
                                 is_split_into_words=True)

    new_labels = []
    for i, labels in enumerate(dataset['ner_tags']):
        labels = [NER_TAG2ID[x] for x in labels]
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs['labels'] = new_labels

    return tokenized_inputs


# ---------------------------------------------------------------------------
def align_labels_with_tokens(labels: List[int],
                             word_ids: List[Optional[int]],
                             cls_token: int = -100) -> List[int]:
    """
    Apply labels to all word indices from the tokenized sequence

    Parameters:
    `labels`: NER labels for the original words in sequence
    `word_ids`: Word indices of tokenized sequence
    `cls_token`: Value to assign for CLS tokens

    Return: Labels for each word index
    """

    label_dict = dict(zip(set(word_ids), [*labels, cls_token]))

    new_labels = [label_dict.get(id, cls_token) for id in word_ids]

    new_labels[1:] = [
        curr + 1 if curr == last and curr % 2 == 1 else curr
        for curr, last in zip(new_labels[1:], new_labels)
    ]

    return new_labels


# ---------------------------------------------------------------------------
def test_align_labels_with_tokens() -> None:
    """ Test align_labels_with_tokens() """

    in_labels = [1, 0, 0]
    word_ids = [None, 0, 0, 0, 1, 2, 2, 2, None]
    out_labels = [-100, 1, 2, 2, 0, 0, 0, 0, -100]

    assert align_labels_with_tokens(in_labels, word_ids) == out_labels

    in_labels = [0, 3, 4, 4, 4, 0, 0]
    word_ids = [None, 0, 1, 2, 3, 4, 4, 5, 6, 6, None]
    out_labels = [-100, 0, 3, 4, 4, 4, 4, 0, 0, 0, -100]

    assert align_labels_with_tokens(in_labels, word_ids) == out_labels


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    sys.exit('This file is a module, and is not meant to be run.')
