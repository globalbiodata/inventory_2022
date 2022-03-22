import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_handler import *
from utils import *


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--checkpoint-filepath',
        type=str,
        default=
        'output_dir/checkpt_biomed_roberta_title_abstract_512_10_epochs',
        help='Location of saved checkpoint file.')
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/val.csv',
        help='Input file. Should contain title/abstract information')
    parser.add_argument(
        '--output_file',
        type=str,
        default='output_dir/predictions.csv',
        help='Output file containing predictions on input_file')
    parser.add_argument(
        '--model-name',
        type=str,
        default='biomed_roberta',
        help=
        "Name of model to try. Can be one of: ['bert', 'biobert', 'scibert', 'pubmedbert', 'pubmedbert_pmc', 'bluebert', 'bluebert_mimic3', 'sapbert', 'sapbert_mean_token', 'bioelectra', 'bioelectra_pmc', 'electramed', 'biomed_roberta', 'biomed_roberta_chemprot', 'biomed_roberta_rct_500']"
    )
    parser.add_argument(
        '--descriptive-labels',
        type=str,
        default=['not-bio-resource', 'bio-resource'],
        help="Descriptive labels corresponding to the [0, 1] numeric scores")
    parser.add_argument('--batch-size', type=int, default=8, help='Batch Size')
    parser.add_argument('--max_len',
                        type=int,
                        default=512,
                        help='Max Sequence Length')
    parser.add_argument(
        '--predictive_field',
        type=str,
        default='title_abstract',
        help=
        "Field in the dataframes to use for prediction. Can be one of ['title', 'abstract', 'title-abstract']"
    )
    parser.add_argument(
        '--labels_field',
        type=str,
        default='curation_score',
        help="Field in the dataframes corresponding to the scores (0, 1)")
    parser.add_argument(
        '--descriptive_labels',
        type=str,
        default=['not-bio-resource', 'bio-resource'],
        help="Descriptive labels corresponding to the [0, 1] numeric scores")

    args, _ = parser.parse_known_args()

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
