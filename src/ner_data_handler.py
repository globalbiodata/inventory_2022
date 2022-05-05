"""
Purpose: Preprocess and tokenize data, create DataLoader
Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import random
import pandas as pd
from typing import List, NamedTuple, Optional, TextIO

from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, DataCollatorForTokenClassification,
                          PreTrainedTokenizer)

from utils import NER_TAG2ID


# ---------------------------------------------------------------------------
class RunParams(NamedTuple):
    """
    Model and run parameters

    `model_name`: Huggingface model name
    `dataset_name`: Dataset split name (train, val, or test)
    `batch_size`: Tokenization batch size
    `num_train`: Number of training datapoints (optional)
    """
    model_name: str
    dataset_name: str
    batch_size: int
    num_train: Optional[int] = None


# ---------------------------------------------------------------------------
def get_dataloader(file: TextIO, run_params: RunParams) -> DataLoader:
    """
    Preprocess data and create dataloader

    Parameters:
    `file`: Input file handle
    `run_params`: Model and run parameters

    Returns:
    A `DataLoader` with preprocessed data
    """

    dataset = load_dataset('pandas',
                           data_files={run_params.dataset_name: file})

    tokenizer = AutoTokenizer.from_pretrained(run_params.model_name)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    tokenized_dataset = tokenize_align_labels(dataset, tokenizer)

    if run_params.num_train:
        tokenized_datasets = tokenized_datasets.select(
            random.sample(range(dataset.num_rows), k=run_params.num_train))

    dataloader = DataLoader(tokenized_dataset,
                            shuffle=True,
                            collate_fn=collator,
                            batch_size=run_params.batch_size)

    return dataloader


# ---------------------------------------------------------------------------
def tokenize_align_labels(dataset: Dataset, tokenizer: PreTrainedTokenizer):

    tokenized_inputs = tokenizer(dataset['words'],
                                 truncation=True,
                                 is_split_into_words=True)

    new_labels = []

    # Should be able to use zip instead of enumerate
    for i, labels in enumerate(dataset['ner_tags']):
        labels = [NER_TAG2ID[x] for x in labels]
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs['labels'] = new_labels

    return tokenized_inputs


# ---------------------------------------------------------------------------
def test_tokenize_align_labels() -> None:
    """ Test tokenize_align_labels() """

    df = pd.DataFrame(
        [[
            123, 0, [0, 1, 2], ['B-COM', 'O', 'O'],
            ['MEGALEX:', 'A', 'megastudy.']
        ],
         [
             123, 1, [0, 1, 2, 3], ['O', 'O', 'B-COM', 'O'],
             ['New', 'database', '(MEGALEX)', 'of.']
         ],
         [
             456, 0, [0, 1, 2, 3, 4, 5],
             ['O', 'B-FUL', 'I-FUL', 'I-FUL', 'I-FUL', 'O'],
             ['The', 'Auditory', 'English', 'Lexicon', 'Project:', 'A.']
         ], [456, 1, [0, 1, 2], ['B-COM', 'O', 'O'], ['(AELP)', 'is', 'a.']]],
        columns=['pmid', 'sent_idx', 'word_idx', 'ner_tags', 'words'])

    dataset = Dataset.from_pandas(df)

    tokenizer = AutoTokenizer.from_pretrained(
        'allenai/scibert_scivocab_uncased')


# ---------------------------------------------------------------------------
def align_labels_with_tokens(labels: List[int],
                             word_ids: List[Optional[int]],
                             cls_token: int = -100) -> List[int]:
    """
    Apply labels to all word indices from the tokenized sequence

    `labels`: NER labels for the original words in sequence
    `word_ids`: Word indices of tokenized sequence
    `cls_token`: Value to assign for CLS tokens

    Return: Labels for each word index
    """

    label_dict = {
        id: label
        for id, label in zip(set(word_ids), [*labels, cls_token])
    }

    is_odd = lambda n: n % 2 == 1

    new_labels = [label_dict.get(id) for id in word_ids]

    new_labels[1:] = [
        curr + 1 if curr == last and is_odd(curr) else curr
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
# class NERDataHandler:
#     """
#     Handles generating training, validation and testing dataloaders used for training and evaluation
#     """
#     def __init__(self,
#                  model_huggingface_version,
#                  batch_size,
#                  train_file,
#                  val_file=None,
#                  test_file=None,
#                  num_train):
#         """
#         :param train_file: path to train file
#         :param val_file: path to val file
#         :param test_file: path to test file
#         :param model_huggingface_version: Hugginface model version used to instantiate the tokenizer
#         """

#         self.tokenizer = AutoTokenizer.from_pretrained(
#             model_huggingface_version)
#         self.dataset = load_dataset('pandas',
#                                     data_files={
#                                         'train': train_file,
#                                         'val': val_file,
#                                         'test': test_file
#                                     })
#         tokenized_datasets = self.dataset.map(
#             self.tokenize_and_align_labels,
#             batched=True,
#             remove_columns=self.dataset["train"].column_names)
#         data_collator = DataCollatorForTokenClassification(
#             tokenizer=self.tokenizer)
#         if sanity_check:
#             tokenized_datasets = tokenized_datasets.select(range(100))
#         self.train_dataloader = DataLoader(tokenized_datasets["train"],
#                                            shuffle=True,
#                                            collate_fn=data_collator,
#                                            batch_size=batch_size)
#         self.val_dataloader = DataLoader(tokenized_datasets["val"],
#                                          collate_fn=data_collator,
#                                          batch_size=batch_size)
#         self.test_dataloader = DataLoader(tokenized_datasets["test"],
#                                           collate_fn=data_collator,
#                                           batch_size=batch_size)

#     def get_dataloaders(self):
#         """
#         Returns train, val and test dataloaders
#         """
#         return self.train_dataloader, self.val_dataloader, self.test_dataloader

#     def align_labels_with_tokens(self, labels, word_ids, cls_token=-100):
#         """
#         Aligns word labels to token labels for NER
#         :param labels: word labels
#         :param word ids: word IDs of tokens
#         :param cls_token: token to assign to CLS tokens
#         :return: labels for tokens
#         """
#         new_labels = []
#         last_word_id = None
#         last_tag = None
#         for word_id in word_ids:
#             if word_id is None:
#                 curr_tag = cls_token
#             elif word_id == last_word_id:
#                 if last_tag % 2 == 1:
#                     curr_tag = last_tag + 1
#                 else:
#                     curr_tag = last_tag
#             else:
#                 curr_tag = labels[word_id]
#                 last_tag = curr_tag
#             new_labels.append(curr_tag)
#             last_word_id = word_id
#         return new_labels

#     def tokenize_and_align_labels(self, examples):
#         """
#         Tokenizes and aligns labels for a given sequence of examples
#         :param examples: sequence to tokenize/assign labels to
#         :return: tokenized inputs
#         """
#         tokenized_inputs = self.tokenizer(examples["words"],
#                                           truncation=True,
#                                           is_split_into_words=True)
#         all_labels = examples["ner_tags"]
#         new_labels = []
#         for i, labels in enumerate(all_labels):
#             labels = [NER_TAG2ID[x] for x in labels]
#             word_ids = tokenized_inputs.word_ids(i)
#             new_labels.append(self.align_labels_with_tokens(labels, word_ids))

#         tokenized_inputs["labels"] = new_labels
#         return tokenized_inputs
