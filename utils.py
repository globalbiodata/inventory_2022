import argparse
from data_handler import *
import torch
from transformers import AutoModelForSequenceClassification

# Mapping from generic model name to the Huggingface Version used to initialize the model
MODEL_TO_HUGGINGFACE_VERSION = {
    'bert': 'bert_base_uncased',
    'biobert': 'dmis-lab/biobert-v1.1',
    'scibert': 'allenai/scibert_scivocab_uncased',
    'pubmedbert': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
    'pubmedbert_pmc':
    'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
    'bluebert': 'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12',
    'bluebert_mimic3': 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
    'sapbert': 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
    'sapbert_mean_token':
    'cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token',
    'bioelectra': 'kamalkraj/bioelectra-base-discriminator-pubmed',
    'bioelectra_pmc': 'kamalkraj/bioelectra-base-discriminator-pubmed-pmc',
    'electramed': 'giacomomiolo/electramed_base_scivocab_1M',
    'biomed_roberta': 'allenai/biomed_roberta_base',
    'biomed_roberta_chemprot':
    'allenai/dsp_roberta_base_dapt_biomed_tapt_chemprot_4169',
    'biomed_roberta_rct_500':
    'allenai/dsp_roberta_base_dapt_biomed_tapt_rct_500'
}