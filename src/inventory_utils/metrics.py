"""
Metrics
~~~

Functions used for calculating and extracting model performance metrics.

Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import sys
from typing import Any, List, Optional, cast

import numpy as np
import torch
from datasets import load_metric
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader

from .custom_classes import Metrics
from .wrangling import convert_to_tags


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
    # pylint: disable=too-many-locals
    calc_seq_metrics = load_metric('seqeval')
    total_loss = 0.
    num_seen_datapoints = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        num_seen_datapoints += len(batch['input_ids'])
        predictions = torch.argmax(outputs.logits, dim=-1)
        predictions_array = predictions.detach().cpu().clone().numpy()
        predictions_array = cast(np.ndarray, predictions)

        labels = cast(Tensor, batch['labels'])
        labels_array = labels.detach().cpu().clone().numpy()
        labels_array = cast(np.ndarray, labels)

        pred_labels, true_labels = convert_to_tags(predictions_array,
                                                   labels_array)

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
if __name__ == '__main__':
    sys.exit('This file is a module, and is not meant to be run.')
