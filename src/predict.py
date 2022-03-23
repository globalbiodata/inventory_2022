import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_handler import DataHandler
from utils import MODEL_TO_HUGGINGFACE_VERSION


# ---------------------------------------------------------------------------
class Predictor():
    """
    Handles prediction based on a trained model
    """
    def __init__(self, model_huggingface_version, checkpoint_filepath):
        """
        """

        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_huggingface_version, num_labels=2)
        checkpoint = torch.load(checkpoint_filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_huggingface_version)
        self.class_labels = ClassLabel(num_classes=2,
                                       names=args.descriptive_labels)

    # -----------------------------------------------------------------------
    def predict(self, dataloader):
        """
  	    Generates predictions for a dataloader containing data
    
  	    :param: dataloader: contains tokenized text
  	    :returns: predicted labels
  	    """
        all_predictions = []
        self.model.eval()
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.extend(predictions)
        predicted_labels = [
            self.class_labels.int2str(int(x)) for x in all_predictions
        ]
        return predicted_labels


# ---------------------------------------------------------------------------
def get_args():
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Predict article classifications using trained BERT model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    inputs = parser.add_argument_group('Inputs and Outputs')
    data_info = parser.add_argument_group('Information on Data')
    model_params = parser.add_argument_group('Model Parameters')
    runtime_params = parser.add_argument_group('Runtime Parameters')

    inputs.add_argument('-c',
                        '--checkpoint-filepath',
                        metavar='CHKPT',
                        type=str,
                        required=True,
                        help='Location of saved checkpoint file.')
    inputs.add_argument(
        '-i',
        '--input_file',
        metavar='FILE',
        type=argparse.FileType('rt'),
        default='data/val.csv',
        help='Input file. Should contain columns in --predictive_field')
    inputs.add_argument('-o',
                        '--output-dir',
                        metavar='DIR',
                        type=str,
                        default='output_dir/',
                        help='Directory to output predictions')
    inputs.add_argument(
        '-of',
        '--output_file',
        metavar='STR',
        type=str,
        default='output_dir/predictions.csv',
        help='Output file containing predictions on input_file')

    data_info.add_argument(
        '-pred',
        '--predictive-field',
        metavar='PRED',
        type=str,
        default='title-abstract',
        help='Field in the dataframes to use for prediction',
        choices=['title', 'abstract', 'title-abstract'])
    data_info.add_argument(
        '-labs',
        '--labels-field',
        metavar='LABS',
        type=str,
        default='curation_score',
        help='Field in the dataframes corresponding to the scores (0, 1)')
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

    return parser.parse_args()


# ---------------------------------------------------------------------------
if __name__ == '__main__':

    args = get_args()

    print(f'args={args}')
    model_huggingface_version = MODEL_TO_HUGGINGFACE_VERSION[args.model_name]
    predictor = Predictor(model_huggingface_version, args.checkpoint_filepath)

    # Load data in a DataLoader
    data_handler = DataHandler(model_huggingface_version, args.input_file)
    data_handler.parse_abstracts_xml()
    data_handler.concatenate_title_abstracts()
    data_handler.generate_dataloaders(args.predictive_field, args.labels_field,
                                      args.descriptive_labels, args.batch_size,
                                      args.max_len)
    dataloader = data_handler.train_dataloader

    # Predict labels
    predicted_labels = predictor.predict(dataloader)
    data_handler.train_df['predicted_label'] = predicted_labels
    pred_df = data_handler.train_df
    true_labels = [
        predictor.class_labels.int2str(int(x))
        for x in data_handler.train_df['curation_score'].values
    ]
    pred_df['true_label'] = true_labels
    pred_df = data_handler.train_df
    print(pred_df[:20])

    # Save labels to file
    pred_df.to_csv(args.output_file)
    print('Saved predictions to', args.output_file)
