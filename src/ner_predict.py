#!/usr/bin/env python3
"""
Purpose: Use trained BERT model for named entity recognition
Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import argparse
import os
import string
from collections import defaultdict
from typing import DefaultDict, Dict, List, NamedTuple, TextIO, Tuple, cast

import pandas as pd
import torch
from pandas.testing import assert_frame_equal
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding, CharSpan

from utils import (ID2NER_TAG, MODEL_TO_HUGGINGFACE_VERSION, NER_TAG2ID,
                   CustomHelpFormatter, get_torch_device, preprocess_data)

pd.options.mode.chained_assignment = None


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    checkpoint: TextIO
    infile: TextIO
    out_dir: str
    model_name: str


# ---------------------------------------------------------------------------
class SeqPrediction(NamedTuple):
    """
    Attributes of predicted sequence labels

    `seq`: Original sequence
    `word_ids`: List of word indices
    `word_locs`: Dictionary giving character spans for each word
    `preds`: Predicted labels
    `probs`: Predicted label probability
    """
    seq: str
    word_ids: List[int]
    word_locs: Dict[int, CharSpan]
    preds: List[str]
    probs: List[float]


# ---------------------------------------------------------------------------
class NamedEntity(NamedTuple):
    """
    Predicted named entity

    `string`: String predicted to be a named entity
    `label`: Predicted label
    `prob`: Probability of predicted label
    """
    string: str
    label: str
    prob: float


# ---------------------------------------------------------------------------
def get_args() -> Args:
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Predict article classifications using trained BERT model',
        formatter_class=CustomHelpFormatter)

    inputs = parser.add_argument_group('Inputs and Outputs')
    model_params = parser.add_argument_group('Model Parameters')
    # runtime_params = parser.add_argument_group('Runtime Parameters')

    inputs.add_argument('-c',
                        '--checkpoint',
                        metavar='CHKPT',
                        type=argparse.FileType('rb'),
                        required=True,
                        help='Trained model checkpoint')
    inputs.add_argument('-i',
                        '--input-file',
                        metavar='FILE',
                        type=argparse.FileType('rt'),
                        required=True,
                        help='Input file for prediction')
    inputs.add_argument('-o',
                        '--out-dir',
                        metavar='DIR',
                        type=str,
                        default='out/',
                        help='Directory to output predictions')

    model_params.add_argument(
        '-m',
        '--model-name',
        metavar='MODEL',
        type=str,
        default='scibert',
        help='Name of model',
        choices=[
            'bert', 'biobert', 'bioelectra', 'bioelectra_pmc',
            'biomed_roberta', 'biomed_roberta_chemprot',
            'biomed_roberta_rct_500', 'bluebert', 'bluebert_mimic3',
            'electramed', 'pubmedbert', 'pubmedbert_pmc', 'sapbert',
            'sapbert_mean_token', 'scibert'
        ])

    args = parser.parse_args()

    return Args(args.checkpoint, args.input_file, args.out_dir,
                args.model_name)


# ---------------------------------------------------------------------------
def get_model(model_name: str, checkpoint_fh: TextIO, device: torch.device):
    """
    Instatiate predictive model from checkpoint

    Params:
    `model_name`: Huggingface model name
    `checkpoint_fh`: Model checkpoint filehandle
    `device`: The `torch.device` to use

    Return:
    Model instance from checkpoint and tokenizer
    """

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, id2label=ID2NER_TAG, label2id=NER_TAG2ID)
    checkpoint = torch.load(checkpoint_fh, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


# ---------------------------------------------------------------------------
def convert_predictions(seq_preds: SeqPrediction) -> List[NamedEntity]:
    """
    Convert raw predictions to meaningful predictions
    """

    seq = seq_preds.seq
    word_locs = seq_preds.word_locs
    mapping: DefaultDict[int, List[Tuple[str, float]]] = defaultdict(list)

    for word_id, label, score in zip(seq_preds.word_ids, seq_preds.preds,
                                     seq_preds.probs):
        mapping[word_id].append((label, score))

    new_mapping = {}
    for word_id, labels_scores in mapping.items():
        start, end = word_locs[word_id]
        word = seq[start:end]
        new_mapping[(
            start, end,
            word)] = labels_scores[0]  # taking first label and probability
    running_start = None
    running_end = None
    running_tag = None
    word2tag = []
    for (start, end, word), tag_pred in new_mapping.items():
        tag = tag_pred[0]
        prob = tag_pred[1]
        if running_end and tag == 'I-' + running_tag[2:]:
            running_end = end
        elif tag[0] == 'B' or tag[0] == 'O':
            if running_start is not None and running_tag != 'O':
                running_word = seq[running_start:running_end]
                entry = NamedEntity(running_word, running_tag, running_pred)
                word2tag.append(entry)
            running_start = start
            running_end = end
            running_tag = tag
            running_pred = prob
    running_word = seq[running_start:running_end]
    if len(running_word) > 0 and running_tag != 'O':
        entry = NamedEntity(running_word, running_tag, running_pred)
        word2tag.append(entry)
    return word2tag


# ---------------------------------------------------------------------------
def test_convert_predictions() -> None:
    """ Test convert_predictions() """

    seq = 'ALCOdb: Gene Coexpression Database for Microalgae.'
    word_ids = [0, 0, 0, 1, 2, 3, 3, 3, 4, 5, 6, 6, 7]
    word_locs = {
        0: CharSpan(0, 6),
        1: CharSpan(6, 7),
        2: CharSpan(8, 12),
        3: CharSpan(13, 25),
        4: CharSpan(26, 34),
        5: CharSpan(35, 38),
        6: CharSpan(39, 49),
        7: CharSpan(49, 50)
    }
    preds = [
        'B-COM', 'I-COM', 'I-COM', 'I-COM', 'B-FUL', 'I-FUL', 'I-FUL', 'I-FUL',
        'I-FUL', 'I-FUL', 'I-FUL', 'I-FUL', 'I-FUL'
    ]
    probs = [
        0.9914267, 0.9947973, 0.9970765, 0.9951375, 0.98841196, 0.9884289,
        0.99392915, 0.9951815, 0.9865631, 0.99616784, 0.99818134, 0.9980192,
        0.90898293
    ]

    seq_preds = SeqPrediction(seq, word_ids, word_locs, preds, probs)

    expected = [
        NamedEntity('ALCOdb:', 'B-COM', 0.9914267),
        NamedEntity('Gene Coexpression Database for Microalgae.', 'B-FUL',
                    0.98841196)
    ]

    assert convert_predictions(seq_preds) == expected


# ---------------------------------------------------------------------------
def predict_sequence(model, device: torch.device, seq: str,
                     tokenizer: PreTrainedTokenizer):

    with torch.no_grad():
        tokenized_seq = tokenizer(seq, return_tensors="pt").to(device)
        outputs = cast(TokenClassifierOutput, model(**tokenized_seq))
        logits = outputs.logits
        preds = logits.argmax(dim=-1).cpu().numpy()[0][1:-1]
        all_probs = torch.nn.functional.softmax(logits,
                                                dim=-1).cpu().numpy()[0][1:-1]
        probs = [prob[pred] for pred, prob in zip(preds, all_probs)]
        labels = [ID2NER_TAG[pred] for pred in preds]
        word_ids = tokenized_seq.word_ids()[1:-1]
        word_locs = {
            id: tokenized_seq.word_to_chars(id)
            for id in set(word_ids)
        }

        seq_preds = SeqPrediction(seq, word_ids, word_locs, labels, probs)

    return convert_predictions(seq_preds)


# ---------------------------------------------------------------------------
def predict(model, tokenizer, inputs: pd.DataFrame, tag_dict: dict,
            device: torch.device):

    all_labels = []
    all_preds = []
    all_IDs = []
    all_texts = []
    all_probs = []
    for _, row in inputs.iterrows():
        id = row['id']
        seq = row['title_abstract']
        predicted_labels = predict_sequence(model, device, seq, tokenizer)
        num_preds = len(predicted_labels)
        mentions = [
            x.string.strip(string.punctuation) for x in predicted_labels
        ]
        labels = [x.label[2:] for x in predicted_labels]
        probs = [x.prob for x in predicted_labels]
        all_labels.extend(labels)
        all_preds.extend(mentions)
        all_IDs.extend([id] * num_preds)
        all_texts.extend([seq] * num_preds)
        all_probs.extend(probs)
    pred_df = pd.DataFrame({
        'ID': all_IDs,
        'text': all_texts,
        'mention': all_preds,
        'label': all_labels,
        'prob': all_probs
    })

    return pred_df


# ---------------------------------------------------------------------------
def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate predicted entities, keeping only highest probability for each
    predicted named entity

    df: Predicted entities dataframe
    """

    unique_df = pd.DataFrame(columns=[*df.columns, 'count'])
    out_df = pd.DataFrame(columns=df.columns)

    # First, remove exact duplicates of a named entity, assigning
    # the highest probability found for that entity
    for _, mention in df.groupby(['ID', 'mention']):
        mention = mention.sort_values('prob', ascending=False)
        mention['count'] = len(mention)
        unique_df = pd.concat([unique_df, mention.head(1)])

    unique_df['uncased_mention'] = unique_df['mention'].str.lower()

    # Remove duplicates that differ only in case
    # Choose which to keep by prioritizing number of occurences then prob
    for _, mention in unique_df.groupby(['ID', 'uncased_mention']):
        mention = mention.sort_values(['count', 'prob'], ascending=False)
        out_df = pd.concat([out_df, mention.head(1)])

    out_df.drop(['count', 'uncased_mention'], axis='columns', inplace=True)
    out_df.reset_index(drop=True, inplace=True)

    return out_df


# ---------------------------------------------------------------------------
def test_deduplicate() -> None:
    """ Test deduplicate() """

    in_df = pd.DataFrame(
        [[
            123, 'SAVI Synthetically Accessible Virtual Inventory', 'SAVI',
            'B-COM', 0.98
        ],
         [
             123, 'SAVI Synthetically Accessible Virtual Inventory',
             'Synthetically Accessible Virtual Inventory', 'B-FUL', 0.64
         ], [456, 'PANTHER PANTHER PANTHER', 'PANTHER', 'B-COM', 0.67],
         [456, 'PANTHER PANTHER PANTHER', 'PANTHER', 'B-COM', 0.95],
         [456, 'PANTHER PANTHER PANTHER', 'PANTHER', 'B-COM', 0.55],
         [789, 'MicrobPad MD (MicrobPad)', 'MicrobPad', 'B-FUL', 0.54],
         [789, 'MicrobPad MD (MicrobPad)', 'MicrobPad', 'B-COM', 0.96],
         [147, 'Chewie-NS Chewie-NS chewie-NS', 'Chewie-NS', 'B-COM', 0.88],
         [147, 'Chewie-NS Chewie-NS chewie-NS', 'Chewie-NS', 'B-COM', 0.72],
         [147, 'Chewie-NS Chewie-NS chewie-NS', 'chewie-NS', 'B-COM', 0.92]],
        columns=['ID', 'text', 'mention', 'label', 'prob'])

    out_df = pd.DataFrame(
        [[
            123, 'SAVI Synthetically Accessible Virtual Inventory', 'SAVI',
            'B-COM', 0.98
        ],
         [
             123, 'SAVI Synthetically Accessible Virtual Inventory',
             'Synthetically Accessible Virtual Inventory', 'B-FUL', 0.64
         ], [147, 'Chewie-NS Chewie-NS chewie-NS', 'Chewie-NS', 'B-COM', 0.88],
         [456, 'PANTHER PANTHER PANTHER', 'PANTHER', 'B-COM', 0.95],
         [789, 'MicrobPad MD (MicrobPad)', 'MicrobPad', 'B-COM', 0.96]],
        columns=['ID', 'text', 'mention', 'label', 'prob'])

    assert_frame_equal(deduplicate(in_df), out_df, check_dtype=False)


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()

    input_df = preprocess_data(args.infile)

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    out_file = os.path.join(args.out_dir, 'predictions.csv')

    model_name = MODEL_TO_HUGGINGFACE_VERSION[args.model_name]

    device = get_torch_device()

    model, tokenizer = get_model(model_name, args.checkpoint, device)

    predictions = deduplicate(
        predict(model, tokenizer, input_df, ID2NER_TAG, device))

    predictions.to_csv(out_file, index=False)

    print(f'Done. Wrote output to {out_file}.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
