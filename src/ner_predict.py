#!/usr/bin/env python3
"""
Purpose: Use trained BERT model for named entity recognition
Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import argparse
import os
import string
from itertools import compress
from statistics import mean
from typing import Dict, List, NamedTuple, TextIO, cast

import pandas as pd
import torch
from pandas.testing import assert_frame_equal
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import CharSpan

from utils import (ID2NER_TAG, NER_TAG2ID, CustomHelpFormatter,
                   get_torch_device, preprocess_data)

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

    model_params.add_argument('-m',
                              '--model-name',
                              metavar='MODEL',
                              type=str,
                              required=True,
                              help='Name of model')

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

    entities: List[NamedEntity] = []
    began_entity = False

    for loc_id, span in seq_preds.word_locs.items():
        mask = [word_id == loc_id for word_id in seq_preds.word_ids]
        labels = set(compress(seq_preds.preds, mask))
        probs = list(compress(seq_preds.probs, mask))
        substring = seq_preds.seq[span.start:span.end + 1]
        if loc_id > 0:
            if seq_preds.word_locs[loc_id - 1].end == span.start:
                substring = seq_preds.seq[span.start + 1:span.end + 1]
        if any(label[0] == 'B' for label in labels):
            began_entity = True
            label = list(
                compress(labels, [label[0] == 'B' for label in labels]))[0]
            entities.append(NamedEntity(substring, label, mean(probs)))
            prob_count = len(probs)
        elif any(label[0] == 'I' for label in labels) and substring != ' ':
            if not began_entity:
                label = list(
                    compress(labels, [label[0] == 'I' for label in labels]))[0]
                entities.append(NamedEntity(substring, label, mean(probs)))
            else:
                last_entity = entities[-1]
                prob = (last_entity.prob * prob_count +
                        sum(probs)) / (prob_count + len(probs))
                prob_count += len(probs)
                entities[-1] = NamedEntity(last_entity.string + substring,
                                           last_entity.label, prob)
        else:
            began_entity = False

    out_entities = []

    for entity in entities:
        out_entities.append(
            NamedEntity(entity.string.rstrip(), entity.label, entity.prob))

    return out_entities


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
        0.9914268, 0.9947973, 0.9970761, 0.9951375, 0.98841196, 0.9884289,
        0.99392915, 0.9951815, 0.9865631, 0.99616784, 0.99818134, 0.9980192,
        0.90898293
    ]

    seq_preds = SeqPrediction(seq, word_ids, word_locs, preds, probs)

    expected = [
        NamedEntity('ALCOdb:', 'B-COM', 0.9944334),
        NamedEntity('Gene Coexpression Database for Microalgae.', 'B-FUL',
                    0.98376288)
    ]

    assert convert_predictions(seq_preds) == expected

    seq = 'Inside outside inside inside (inside).'
    word_ids = [0, 1, 2, 3, 4, 5, 6]
    word_locs = {
        0: CharSpan(0, 6),
        1: CharSpan(7, 14),
        2: CharSpan(15, 21),
        3: CharSpan(22, 28),
        4: CharSpan(29, 30),
        5: CharSpan(30, 36),
        6: CharSpan(36, 37)
    }
    preds = ['I-COM', 'O', 'I-FUL', 'I-FUL', 'B-COM', 'I-COM', 'I-COM']
    probs = [0.996, 0.999, 0.998, 0.978, 0.99, 0.98, 0.97]

    seq_preds = SeqPrediction(seq, word_ids, word_locs, preds, probs)

    expected = [
        NamedEntity('Inside', 'I-COM', 0.996),
        NamedEntity('inside', 'I-FUL', 0.998),
        NamedEntity('inside', 'I-FUL', 0.978),
        NamedEntity('(inside).', 'B-COM', 0.98)
    ]

    assert convert_predictions(seq_preds) == expected


# ---------------------------------------------------------------------------
def predict_sequence(model, device: torch.device, seq: str,
                     tokenizer: PreTrainedTokenizer) -> List[NamedEntity]:
    """
    Run token prediction on sequence

    `model`: Trained token classification model
    `device`: Device to use
    `seq`: Input string/sequence
    `tokenizer`: Pretrained tokenizer

    Return list of named entities
    """

    with torch.no_grad():
        tokenized_seq = tokenizer(seq,
                                  return_tensors="pt",
                                  padding=True,
                                  truncation=True,
                                  max_length=512).to(device)
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
def predict(model, tokenizer: PreTrainedTokenizer, inputs: pd.DataFrame,
            device: torch.device) -> pd.DataFrame:
    """
    Perform NER prediction on rows of input dataframe

    `model`: Trained token classification model
    `tokenizer`: Pretrained tokenizer
    `inputs`: Input dataframe
    `device`: Device to use

    return
    """

    pred_df = pd.DataFrame(columns=['ID', 'text', 'mention', 'label', 'prob'])

    for _, row in inputs.iterrows():
        seq = row['title_abstract']
        predicted_labels = predict_sequence(model, device, seq, tokenizer)
        num_preds = len(predicted_labels)
        mentions = [
            x.string.strip(string.punctuation) for x in predicted_labels
        ]
        labels = [x.label[2:] for x in predicted_labels]
        probs = [x.prob for x in predicted_labels]
        pred_df = pd.concat([
            pred_df,
            pd.DataFrame({
                'ID': [row['id']] * num_preds,
                'text': [seq] * num_preds,
                'mention': mentions,
                'label': labels,
                'prob': probs
            })
        ])

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
            'COM', 0.98
        ],
         [
             123, 'SAVI Synthetically Accessible Virtual Inventory',
             'Synthetically Accessible Virtual Inventory', 'FUL', 0.64
         ], [456, 'PANTHER PANTHER PANTHER', 'PANTHER', 'COM', 0.67],
         [456, 'PANTHER PANTHER PANTHER', 'PANTHER', 'COM', 0.95],
         [456, 'PANTHER PANTHER PANTHER', 'PANTHER', 'COM', 0.55],
         [789, 'MicrobPad MD (MicrobPad)', 'MicrobPad', 'FUL', 0.54],
         [789, 'MicrobPad MD (MicrobPad)', 'MicrobPad', 'COM', 0.96],
         [147, 'Chewie-NS Chewie-NS chewie-NS', 'Chewie-NS', 'COM', 0.88],
         [147, 'Chewie-NS Chewie-NS chewie-NS', 'Chewie-NS', 'COM', 0.72],
         [147, 'Chewie-NS Chewie-NS chewie-NS', 'chewie-NS', 'COM', 0.92]],
        columns=['ID', 'text', 'mention', 'label', 'prob'])

    out_df = pd.DataFrame(
        [[
            123, 'SAVI Synthetically Accessible Virtual Inventory', 'SAVI',
            'COM', 0.98
        ],
         [
             123, 'SAVI Synthetically Accessible Virtual Inventory',
             'Synthetically Accessible Virtual Inventory', 'FUL', 0.64
         ], [147, 'Chewie-NS Chewie-NS chewie-NS', 'Chewie-NS', 'COM', 0.88],
         [456, 'PANTHER PANTHER PANTHER', 'PANTHER', 'COM', 0.95],
         [789, 'MicrobPad MD (MicrobPad)', 'MicrobPad', 'COM', 0.96]],
        columns=['ID', 'text', 'mention', 'label', 'prob'])

    assert_frame_equal(deduplicate(in_df), out_df, check_dtype=False)


# ---------------------------------------------------------------------------
def reformat_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reformat output datframe to wide format

    `df`: Dataframe output by deduplicate()

    Return
    Wide-format datframe
    """

    df['prob'] = df['prob'].astype(str)

    # Add two dummy rows so that both COM and FUL are present as labels
    df.loc[len(df)] = ['-1', '', '', 'COM', '0']
    df.loc[len(df)] = ['-1', '', '', 'FUL', '0']
    df = df[df['mention'] != '']

    # For each article, aggregate multiple occurences
    # of same label into single row
    df2 = df['mention'].groupby([df.ID, df.text,
                                 df.label]).apply(list).reset_index()
    df2['prob'] = df['prob'].groupby([df.ID, df.text, df.label
                                      ]).apply(list).reset_index()['prob']

    # Create combined column of mentions and their probs
    df2['mention_prob'] = list(zip(df2['mention'], df2['prob']))

    # Pivot to wide format, each label gets its own column
    df2 = df2.pivot(index=['ID', 'text'],
                    columns='label',
                    values='mention_prob')

    # Fill missing values
    for col in [c for c in list(df2.columns) if c not in ['ID', 'text']]:
        isna = df2[col].isna()
        df2.loc[isna, col] = pd.Series([[[''], ['']]] * isna.sum(),
                                       dtype='object').values

    # Split mentions and probs to their own columns
    # and drop the unsplit columns
    df2[['common_name', 'common_prob']] = pd.DataFrame(df2['COM'].tolist(),
                                                       index=df2.index)
    df2[['full_name', 'full_prob']] = pd.DataFrame(df2['FUL'].tolist(),
                                                   index=df2.index)
    df2.drop(['COM', 'FUL'], inplace=True, axis='columns')

    # Convert lists of multiple mentions to string with commas between
    for col in [c for c in list(df2.columns) if c not in ['ID', 'text']]:
        df2[col] = df2[col].fillna('')
        df2[col] = df2[col].apply(', '.join)

    df2.reset_index(inplace=True)

    # Remove the dummy row
    df2 = df2[df2['ID'] != '-1']

    return df2


# ---------------------------------------------------------------------------
def test_reformat_output() -> None:
    """ Test reformat_output() """

    in_df = pd.DataFrame(
        [[
            123, 'SAVI Synthetically Accessible Virtual Inventory', 'SAVI',
            'COM', 0.98
        ],
         [
             123, 'SAVI Synthetically Accessible Virtual Inventory',
             'Synthetically Accessible Virtual Inventory', 'FUL', 0.64
         ], [147, 'Chewie-NS Chewie-NS chewie-NS', 'Chewie-NS', 'COM', 0.88],
         [456, 'PANTHER PANTHER LION', 'PANTHER', 'COM', 0.95],
         [456, 'PANTHER PANTHER LION', 'LION', 'COM', 0.92],
         [789, 'MicrobPad MD (MicrobPad)', 'MicrobPad', 'COM', 0.96]],
        columns=['ID', 'text', 'mention', 'label', 'prob'])

    out_df = pd.DataFrame(
        [[
            123, 'SAVI Synthetically Accessible Virtual Inventory', 'SAVI',
            '0.98', 'Synthetically Accessible Virtual Inventory', '0.64'
        ], [147, 'Chewie-NS Chewie-NS chewie-NS', 'Chewie-NS', '0.88', '', ''],
         [456, 'PANTHER PANTHER LION', 'PANTHER, LION', '0.95, 0.92', '', ''],
         [789, 'MicrobPad MD (MicrobPad)', 'MicrobPad', '0.96', '', '']],
        columns=[
            'ID', 'text', 'common_name', 'common_prob', 'full_name',
            'full_prob'
        ])

    assert_frame_equal(reformat_output(in_df), out_df, check_names=False)


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()

    input_df = preprocess_data(args.infile)

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    out_file = os.path.join(args.out_dir, 'predictions.csv')

    model_name = args.model_name

    device = get_torch_device()

    model, tokenizer = get_model(model_name, args.checkpoint, device)

    predictions = reformat_output(
        deduplicate(predict(model, tokenizer, input_df, device)))

    predictions.to_csv(out_file, index=False)

    print(f'Done. Saved predictions to {out_file}.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
