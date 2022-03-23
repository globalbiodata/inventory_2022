#!/usr/bin/env python3
"""
Purpose: Use trained BERT model for article classification
Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import argparse
import os
from typing import List, NamedTuple, TextIO

import torch
from datasets import ClassLabel
from torch.utils.data.dataloader import DataLoader
from transformers import AutoModelForSequenceClassification as classifier

from data_handler import DataHandler
from utils import MODEL_TO_HUGGINGFACE_VERSION


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    checkpoint: TextIO
    infile: TextIO
    out_dir: str
    out_file: str
    predictive_field: str
    labels_field: str
    descriptive_labels: str
    model_name: str
    max_len: int
    batch_size: int


# ---------------------------------------------------------------------------
def get_args() -> Args:
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Predict article classifications using trained BERT model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    inputs = parser.add_argument_group('Inputs and Outputs')
    data_info = parser.add_argument_group('Information on Data')
    model_params = parser.add_argument_group('Model Parameters')
    runtime_params = parser.add_argument_group('Runtime Parameters')

    inputs.add_argument('-c',
                        '--checkpoint',
                        metavar='CHKPT',
                        type=argparse.FileType('rb'),
                        required=True,
                        help='Trained model checkpoint')
    inputs.add_argument(
        '-i',
        '--input_file',
        metavar='FILE',
        type=argparse.FileType('rt'),
        default='data/val.csv',
        help='Input file. Should contain columns in --predictive_field')
    inputs.add_argument('-o',
                        '--out-dir',
                        metavar='DIR',
                        type=str,
                        default='output_dir/',
                        help='Directory to output predictions')
    inputs.add_argument(
        '-of',
        '--out-file',
        metavar='STR',
        type=str,
        default='predictions.csv',
        help='Output file containing predictions on --input_file')

    data_info.add_argument(
        '-pred',
        '--predictive-field',
        metavar='PRED',
        type=str,
        default='title_abstract',
        help='Field in the dataframes to use for prediction',
        choices=['title', 'abstract', 'title_abstract'])
    data_info.add_argument(
        '-desc',
        '--descriptive-labels',
        metavar='LAB',
        type=str,
        nargs=2,
        default=['not-bio-resource', 'bio-resource'],
        help='Descriptive labels corresponding to the [0, 1] numeric scores')

    model_params.add_argument(
        '-m',
        '--model-name',
        metavar='MODEL',
        type=str,
        default='scibert',
        help='Name of model',
        choices=[
            'bert', 'biobert', 'scibert', 'pubmedbert', 'pubmedbert_pmc',
            'bluebert', 'bluebert_mimic3', 'sapbert', 'sapbert_mean_token',
            'bioelectra', 'bioelectra_pmc', 'electramed', 'biomed_roberta',
            'biomed_roberta_chemprot', 'biomed_roberta_rct_500'
        ])
    model_params.add_argument('-max',
                              '--max-len',
                              metavar='INT',
                              type=int,
                              default=256,
                              help='Max Sequence Length')

    runtime_params.add_argument('-batch',
                                '--batch-size',
                                metavar='INT',
                                type=int,
                                default=8,
                                help='Batch Size')

    args = parser.parse_args()

    return Args(args.checkpoint, args.input_file, args.out_dir, args.out_file,
                args.predictive_field, args.labels_field,
                args.descriptive_labels, args.model_name, args.max_len,
                args.batch_size)


# ---------------------------------------------------------------------------
def get_datahandler(model_name: str, args: Args) -> DataHandler:
    """ Generate the dataloader """

    data_handler = DataHandler(model_name, args.infile)
    data_handler.parse_abstracts_xml()
    data_handler.concatenate_title_abstracts()
    data_handler.generate_dataloaders(args.predictive_field, args.labels_field,
                                      args.descriptive_labels, args.batch_size,
                                      args.max_len)

    return data_handler


# ---------------------------------------------------------------------------
def get_torch_device() -> torch.device:
    """ Get device for torch """

    return torch.device('cuda') if torch.cuda.is_available() else torch.device(
        'cpu')


# ---------------------------------------------------------------------------
def get_model(model_name: str, checkpoint_fh: TextIO, device: torch.device):
    """ Instatiate predictive model from checkpoint """

    model = classifier.from_pretrained(model_name, num_labels=2)
    checkpoint = torch.load(checkpoint_fh, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


# ---------------------------------------------------------------------------
def predict(model, dataloader: DataLoader, class_labels: ClassLabel,
            device: torch.device) -> List[str]:
    """ Use model to predict article classifications """

    all_predictions = []
    model.eval()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        all_predictions.extend(predictions)

    predicted_labels = [class_labels.int2str(int(x)) for x in all_predictions]

    return predicted_labels


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    out_file = os.path.join(args.out_dir, args.out_file)

    model_name = MODEL_TO_HUGGINGFACE_VERSION[args.model_name]

    data_handler = get_datahandler(model_name, args)
    dataloader = data_handler.train_dataloader

    device = get_torch_device()

    model = get_model(model_name, args.checkpoint, device)
    class_labels = ClassLabel(num_classes=2, names=args.descriptive_labels)

    # Predict labels
    predicted_labels = predict(model, dataloader, class_labels, device)
    data_handler.train_df['predicted_label'] = predicted_labels
    pred_df = data_handler.train_df

    # Save labels to file
    pred_df.to_csv(out_file)
    print('Saved predictions to', out_file)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
