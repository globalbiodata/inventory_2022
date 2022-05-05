"""
Purpose: Preprocess and tokenize data, create DataLoader
Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import random
from typing import NamedTuple, Optional, TextIO

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, DataCollatorForTokenClassification,
                          PreTrainedTokenizer)

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
def get_dataloader(file: TextIO, run_params: RunParams) -> DataLoader:
    """
    Preprocess data and create dataloader

    Parameters:
    `file`: Input file handle
    `run_params`: Model and run parameters

    Returns:
    A `DataLoader` with preprocessed data
    """

    dataset = load_dataset('pandas', data_files=file)

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
def tokenize_align_labels(dataset, tokenizer: PreTrainedTokenizer):

    tokenized_inputs = tokenizer(dataset['words'],
                                 truncation=True,
                                 is_split_into_words=True)

    new_labels = []

    # Should be able to use zip instead of enumerate
    for i, labels in enumerate(dataset['words']):
        labels = [NER_TAG2ID[x] for x in labels]
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs['labels'] = new_labels

    return tokenized_inputs


# ---------------------------------------------------------------------------
def align_labels_with_tokens(labels, word_ids, cls_token=-100):

    new_labels = []
    last_word_id = None
    last_tag = None
    for word_id in word_ids:
        if word_id is None:
            curr_tag = cls_token
        elif word_id == last_word_id:
            if last_tag % 2 == 1:
                curr_tag = last_tag + 1
            else:
                curr_tag = last_tag
        else:
            curr_tag = labels[word_id]
            last_tag = curr_tag
        new_labels.append(curr_tag)
        last_word_id = word_id

    return new_labels


# ---------------------------------------------------------------------------
class NERDataHandler:
    """
    Handles generating training, validation and testing dataloaders used for training and evaluation
    """
    def __init__(self,
                 model_huggingface_version,
                 batch_size,
                 train_file,
                 val_file=None,
                 test_file=None,
                 num_train):
        """
        :param train_file: path to train file
        :param val_file: path to val file
        :param test_file: path to test file
        :param model_huggingface_version: Hugginface model version used to instantiate the tokenizer
        """

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_huggingface_version)
        self.dataset = load_dataset('pandas',
                                    data_files={
                                        'train': train_file,
                                        'val': val_file,
                                        'test': test_file
                                    })
        tokenized_datasets = self.dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=self.dataset["train"].column_names)
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer)
        if sanity_check:
            tokenized_datasets = tokenized_datasets.select(range(100))
        self.train_dataloader = DataLoader(tokenized_datasets["train"],
                                           shuffle=True,
                                           collate_fn=data_collator,
                                           batch_size=batch_size)
        self.val_dataloader = DataLoader(tokenized_datasets["val"],
                                         collate_fn=data_collator,
                                         batch_size=batch_size)
        self.test_dataloader = DataLoader(tokenized_datasets["test"],
                                          collate_fn=data_collator,
                                          batch_size=batch_size)

    def get_dataloaders(self):
        """
        Returns train, val and test dataloaders
        """
        return self.train_dataloader, self.val_dataloader, self.test_dataloader

    def align_labels_with_tokens(self, labels, word_ids, cls_token=-100):
        """
        Aligns word labels to token labels for NER
        :param labels: word labels
        :param word ids: word IDs of tokens
        :param cls_token: token to assign to CLS tokens
        :return: labels for tokens
        """
        new_labels = []
        last_word_id = None
        last_tag = None
        for word_id in word_ids:
            if word_id is None:
                curr_tag = cls_token
            elif word_id == last_word_id:
                if last_tag % 2 == 1:
                    curr_tag = last_tag + 1
                else:
                    curr_tag = last_tag
            else:
                curr_tag = labels[word_id]
                last_tag = curr_tag
            new_labels.append(curr_tag)
            last_word_id = word_id
        return new_labels

    def tokenize_and_align_labels(self, examples):
        """
        Tokenizes and aligns labels for a given sequence of examples
        :param examples: sequence to tokenize/assign labels to
        :return: tokenized inputs
        """
        tokenized_inputs = self.tokenizer(examples["words"],
                                          truncation=True,
                                          is_split_into_words=True)
        all_labels = examples["ner_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            labels = [NER_TAG2ID[x] for x in labels]
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs
