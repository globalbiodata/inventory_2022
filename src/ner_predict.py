#!/usr/bin/env python3
"""
Purpose: Use trained BERT model for named entity recognition
Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import argparse
import os
import string
from typing import NamedTuple, TextIO

import pandas as pd
import torch
from pandas.testing import assert_frame_equal
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

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
def predict_sequence(model, device: torch.device, seq: str,
                     tokenizer: PreTrainedTokenizer):

    with torch.no_grad():
        tokenized_seq = tokenizer(seq, return_tensors="pt").to(device)
        outputs = model(**tokenized_seq)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(
            outputs.logits, dim=-1).cpu().numpy()[0][1:-1]
        predictions = logits.argmax(dim=-1).cpu().numpy()[0][1:-1]
        word_ids = tokenized_seq.word_ids()[1:-1]
        mapping = {}

        for word_id, pred, prob in zip(word_ids, predictions, probabilities):
            label = ID2NER_TAG[pred]
            score = prob[pred]
            if word_id in mapping:
                mapping[word_id].append((label, score))
            else:
                mapping[word_id] = [(label, score)]
        new_mapping = {}
        for word_id, labels_scores in mapping.items():
            start, end = tokenized_seq.word_to_chars(word_id)
            word = seq[start:end]
            new_mapping[(start, end, word)] = labels_scores[0]
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
                    entry = {
                        'label': running_tag,
                        'prob': running_pred,
                        'word': running_word  #,
                        # 'start': running_start,
                        # 'end': running_end
                    }
                    word2tag.append(entry)
                running_start = start
                running_end = end
                running_tag = tag
                running_pred = prob
        running_word = seq[running_start:running_end]
        if len(running_word) > 0 and running_tag != 'O':
            entry = {
                'label': running_tag,
                'prob': running_pred,
                'word': running_word  #,
                # 'start': running_start,
                # 'end': running_end
            }
            word2tag.append(entry)
    return word2tag


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
            x['word'].strip(string.punctuation) for x in predicted_labels
        ]
        labels = [x['label'][2:] for x in predicted_labels]
        probs = [x['prob'] for x in predicted_labels]
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

    out_df = pd.DataFrame(columns=df.columns)

    for _, mention in df.groupby(['ID', 'mention']):
        mention = mention.reset_index(drop=True).sort_values('prob',
                                                             ascending=False)
        out_df = pd.concat([out_df, mention.head(1)])

    out_df.reset_index(inplace=True, drop=True)

    return out_df


# ---------------------------------------------------------------------------
def test_deduplicate() -> None:
    """ Test deduplicate() """

    in_df = pd.DataFrame([
        [
            123, 'SAVI Synthetically Accessible Virtual Inventory', 'SAVI',
            'B-COM', 0.98
        ],
        [
            123, 'SAVI Synthetically Accessible Virtual Inventory',
            'Synthetically Accessible Virtual Inventory', 'B-FUL', 0.64
        ],
        [456, 'PANTHER PANTHER PANTHER', 'PANTHER', 'B-COM', 0.67],
        [456, 'PANTHER PANTHER PANTHER', 'PANTHER', 'B-COM', 0.95],
        [456, 'PANTHER PANTHER PANTHER', 'PANTHER', 'B-COM', 0.55],
        [789, 'MicrobPad MD (MicrobPad)', 'MicrobPad', 'B-FUL', 0.54],
        [789, 'MicrobPad MD (MicrobPad)', 'MicrobPad', 'B-COM', 0.96],
    ],
                         columns=['ID', 'text', 'mention', 'label', 'prob'])

    out_df = pd.DataFrame([
        [
            123, 'SAVI Synthetically Accessible Virtual Inventory', 'SAVI',
            'B-COM', 0.98
        ],
        [
            123, 'SAVI Synthetically Accessible Virtual Inventory',
            'Synthetically Accessible Virtual Inventory', 'B-FUL', 0.64
        ],
        [456, 'PANTHER PANTHER PANTHER', 'PANTHER', 'B-COM', 0.95],
        [789, 'MicrobPad MD (MicrobPad)', 'MicrobPad', 'B-COM', 0.96],
    ],
                          columns=['ID', 'text', 'mention', 'label', 'prob'])

    assert_frame_equal(deduplicate(in_df), out_df, check_dtype=False)


# # ---------------------------------------------------------------------------
# class NERPredictor():
#     """
#   Handles prediction based on a trained model
#   """
#     def __init__(self, model_huggingface_version, checkpoint_filepath):
#         """
#     :param model_huggingface_version: HuggingFace model version to load the pretrained model weights from
#     :param checkpoint_filepath: saved checkpt to load the model from
#     """
# self.device = torch.device(
#     "cuda") if torch.cuda.is_available() else torch.device("cpu")
# self.model = AutoModelForTokenClassification.from_pretrained(
#     model_huggingface_version,
#     id2label=ID2NER_TAG,
#     label2id=NER_TAG2ID)
# checkpoint = torch.load(checkpoint_filepath, map_location=self.device)
# self.model.load_state_dict(checkpoint['model_state_dict'])
# self.model.to(self.device)
# self.model.eval()
# self.tokenizer = AutoTokenizer.from_pretrained(
#     model_huggingface_version)

# def predict(self, text):
#     """
# Generates predictions for a sentence using the trained model

# :returns: predicted labels
# """
#     with torch.no_grad():
#         inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
#         outputs = self.model(**inputs)
#         logits = outputs.logits
#         probabilities = torch.nn.functional.softmax(
#             outputs.logits, dim=-1).cpu().numpy()[0][1:-1]
#         predictions = logits.argmax(dim=-1).cpu().numpy()[0][1:-1]
#         word_ids = inputs.word_ids()[1:-1]
#         mapping = {}

#         for word_id, pred, prob in zip(word_ids, predictions,
#                                        probabilities):
#             label = ID2NER_TAG[pred]
#             score = prob[pred]
#             if word_id in mapping:
#                 mapping[word_id].append((label, score))
#             else:
#                 mapping[word_id] = [(label, score)]
#         new_mapping = {}
#         for word_id, labels_scores in mapping.items():
#             start, end = inputs.word_to_chars(word_id)
#             word = text[start:end]
#             new_mapping[(start, end, word)] = labels_scores[0]
#         running_start = None
#         running_end = None
#         running_tag = None
#         word2tag = []
#         for (start, end, word), tag_pred in new_mapping.items():
#             tag = tag_pred[0]
#             prob = tag_pred[1]
#             if running_end and tag == 'I-' + running_tag[2:]:
#                 running_end = end
#             elif tag[0] == 'B' or tag[0] == 'O':
#                 if running_start is not None and running_tag != 'O':
#                     running_word = text[running_start:running_end]
#                     entry = {
#                         'label': running_tag,
#                         'prob': running_pred,
#                         'word': running_word,
#                         'start': running_start,
#                         'end': running_end
#                     }
#                     word2tag.append(entry)
#                 running_start = start
#                 running_end = end
#                 running_tag = tag
#                 running_pred = prob
#         running_word = text[running_start:running_end]
#         if len(running_word) > 0 and running_tag != 'O':
#             entry = {
#                 'label': running_tag,
#                 'prob': running_pred,
#                 'word': running_word,
#                 'start': running_start,
#                 'end': running_end
#             }
#             word2tag.append(entry)
#             print(word2tag)
#     return word2tag


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

    print(predictions[:20])


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()

    # Predict labels
    # predictor = NERPredictor(model_huggingface_version,
    #                          args.checkpoint_filepath)
    # all_preds = []
    # all_IDs = []
    # all_texts = []
    # all_probs = []
    # all_offsets_start = []
    # all_offsets_end = []
    # for ID, text in zip(IDs, text_arr):
    #     predicted_labels = predictor.predict(text)
    #     num_preds = len(predicted_labels)
    #     mentions = [x['word'] for x in predicted_labels]
    #     probs = [x['prob'] for x in predicted_labels]
    #     offsets_start = [x['start'] for x in predicted_labels]
    #     offsets_end = [x['end'] for x in predicted_labels]
    #     all_preds.extend(mentions)
    #     all_IDs.extend([ID] * num_preds)
    #     all_texts.extend([text] * num_preds)
    #     all_probs.extend(probs)
    #     all_offsets_start.extend(offsets_start)
    #     all_offsets_end.extend(offsets_end)
    # pred_df = pd.DataFrame({
    #     'ID': all_IDs,
    #     'text': all_texts,
    #     'mention': all_preds,
    #     'prob': all_probs,
    #     'start_offset': all_offsets_start,
    #     'end_offset': all_offsets_end
    # })
    # print(pred_df[:20])

    # # Save labels to file
    # pred_df.to_csv(args.output_file)
    # print('Saved predictions to', args.output_file)
