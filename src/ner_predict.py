#!/usr/bin/env python3
"""
Purpose: Perform named entity recognition
Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import argparse

import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from utils import ID2NER_TAG, MODEL_TO_HUGGINGFACE_VERSION, NER_TAG2ID


class NERPredictor():
    """
  Handles prediction based on a trained model
  """
    def __init__(self, model_huggingface_version, checkpoint_filepath):
        """
    :param model_huggingface_version: HuggingFace model version to load the pretrained model weights from
    :param checkpoint_filepath: saved checkpt to load the model from
    """
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_huggingface_version,
            id2label=ID2NER_TAG,
            label2id=NER_TAG2ID)
        checkpoint = torch.load(checkpoint_filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_huggingface_version)

    def predict(self, text):
        """
    Generates predictions for a sentence using the trained model

    :returns: predicted labels
    """
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(
                outputs.logits, dim=-1).cpu().numpy()[0][1:-1]
            predictions = logits.argmax(dim=-1).cpu().numpy()[0][1:-1]
            word_ids = inputs.word_ids()[1:-1]
            mapping = {}

            for word_id, pred, prob in zip(word_ids, predictions,
                                           probabilities):
                label = ID2NER_TAG[pred]
                score = prob[pred]
                if word_id in mapping:
                    mapping[word_id].append((label, score))
                else:
                    mapping[word_id] = [(label, score)]
            new_mapping = {}
            for word_id, labels_scores in mapping.items():
                start, end = inputs.word_to_chars(word_id)
                word = text[start:end]
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
                        running_word = text[running_start:running_end]
                        entry = {
                            'label': running_tag,
                            'prob': running_pred,
                            'word': running_word,
                            'start': running_start,
                            'end': running_end
                        }
                        word2tag.append(entry)
                    running_start = start
                    running_end = end
                    running_tag = tag
                    running_pred = prob
            running_word = text[running_start:running_end]
            if len(running_word) > 0 and running_tag != 'O':
                entry = {
                    'label': running_tag,
                    'prob': running_pred,
                    'word': running_word,
                    'start': running_start,
                    'end': running_end
                }
                word2tag.append(entry)
                print(word2tag)
        return word2tag


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint-filepath',
                        type=str,
                        default='checkpts/checkpt_ner_biomed_roberta_2_epochs',
                        help='Location of saved checkpoint file.')
    parser.add_argument(
        '--output_file',
        type=str,
        default='output_dir/preds_ner.csv',
        help='Output file containing predictions on input_file')
    parser.add_argument(
        '--model-name',
        type=str,
        default='biomed_roberta',
        help=
        "Name of model to try. Can be one of: ['bert', 'biobert', 'scibert', 'pubmedbert', 'pubmedbert_fulltext', 'bluebert', 'bluebert_mimic3', 'sapbert', 'sapbert_mean_token', 'bioelectra', 'bioelectra_pmc', 'electramed', 'biomed_roberta', 'biomed_roberta_chemprot', 'biomed_roberta_rct500']"
    )

    args, _ = parser.parse_known_args()

    print(f'args={args}')
    model_huggingface_version = MODEL_TO_HUGGINGFACE_VERSION[args.model_name]

    # Example running on sentences
    text1 = "Information from structural genomics experiments at the RIKEN SPring-8 Center, Japan has been compiled and published as an /integrated database. The contents of the database are (i) experimental data from nine species of bacteria that cover a large variety of protein molecules in terms of both evolution and properties (http://database.riken.jp/db/bacpedia), (ii) experimental data from mutant proteins that were designed systematically to study the influence of mutations on the diffraction quality of protein crystals (http://database.riken.jp/db/bacpedia) and (iii) experimental data from heavy-atom-labelled proteins from the heavy-atom database HATODAS (http://database.riken.jp/db/hatodas). The database integration adopts the semantic web, which is suitable for data reuse and automatic processing, thereby allowing batch downloads of full data and data reconstruction to produce new databases. In addition, to enhance the use of data (i) and (ii) by general researchers in biosciences, a comprehensible user interface, Bacpedia (http://bacpedia.harima.riken.jp), has been developed."
    text2 = "DPL (http://www.peptide-ligand.cn/) is a comprehensive database of peptide ligand (DPL). DPL1.0 holds 1044 peptide ligand entries and provides references for the study of the polypeptide platform. The data were collected from PubMed-NCBI, PDB, APD3, CAMPR3, etc. The lengths of the base sequences are varied from 3 to78. DPL database has 923 linear peptides and 88 cyclic peptides. The functions of peptides collected by DPL are very wide. It includes 540 entries of antiviral peptides (including SARS-CoV-2), 55 entries of signal peptides, 48 entries of protease inhibitors, 45 entries of anti-hypertension, 37 entries of anticancer peptides, etc. There are 270 different kinds of peptide targets. All peptides in DPL have clear binding targets. Most of the peptides and receptors have 3D structures experimentally verified or predicted by CYCLOPS, I-TASSER and SWISS-MODEL. With the rapid development of the COVID-2019 epidemic, this database also collects the research progress of peptides against coronavirus. In conclusion, DPL is a unique resource, which allows users easily to explore the targets, different structures as well as properties of peptides."

    IDs = ['text1', 'text2']
    text_arr = [text1, text2]

    # Predict labels
    predictor = NERPredictor(model_huggingface_version,
                             args.checkpoint_filepath)
    all_preds = []
    all_IDs = []
    all_texts = []
    all_probs = []
    all_offsets_start = []
    all_offsets_end = []
    for ID, text in zip(IDs, text_arr):
        predicted_labels = predictor.predict(text)
        num_preds = len(predicted_labels)
        mentions = [x['word'] for x in predicted_labels]
        probs = [x['prob'] for x in predicted_labels]
        offsets_start = [x['start'] for x in predicted_labels]
        offsets_end = [x['end'] for x in predicted_labels]
        all_preds.extend(mentions)
        all_IDs.extend([ID] * num_preds)
        all_texts.extend([text] * num_preds)
        all_probs.extend(probs)
        all_offsets_start.extend(offsets_start)
        all_offsets_end.extend(offsets_end)
    pred_df = pd.DataFrame({
        'ID': all_IDs,
        'text': all_texts,
        'mention': all_preds,
        'prob': all_probs,
        'start_offset': all_offsets_start,
        'end_offset': all_offsets_end
    })
    print(pred_df[:20])

    # Save labels to file
    pred_df.to_csv(args.output_file)
    print('Saved predictions to', args.output_file)
