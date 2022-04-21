"""
Purpose: Provide shared data structures and classes
Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import argparse
import re
from typing import List, NamedTuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Mapping from generic model name to the Huggingface Version
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

# ---------------------------------------------------------------------------
# Hyperparameters used for training
ARGS_MAP = {
    'bert': ['bert-base-uncased', 16, 3e-5, 0, False],
    'biobert': ['dmis-lab/biobert-v1.1', 16, '3e-5', 0, False],
    'scibert': ['allenai/scibert_scivocab_uncased', 16, 3e-5, 0, False],
    'pubmedbert': [
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', 16, 3e-5, 0,
        True
    ],
    'pubmedbert_fulltext': [
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', 32,
        3e-5, 0, True
    ],
    'bluebert':
    ['bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12', 16, 3e-5, 0, True],
    'bluebert_mimic3': [
        'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12', 32, 3e-5, 0,
        False
    ],
    'sapbert':
    ['cambridgeltl/SapBERT-from-PubMedBERT-fulltext', 16, 2e-5, 0.01, False],
    'sapbert_mean_token': [
        'cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token', 32, 2e-5,
        0.01, False
    ],
    'bioelectra':
    ['kamalkraj/bioelectra-base-discriminator-pubmed', 16, 5e-5, 0, True],
    'bioelectra_pmc':
    ['kamalkraj/bioelectra-base-discriminator-pubmed-pmc', 32, 5e-5, 0, True],
    'electramed':
    ['giacomomiolo/electramed_base_scivocab_1M', 16, 5e-5, 0, True],
    'biomed_roberta': ['allenai/biomed_roberta_base', 16, 2e-5, 0, False],
    'biomed_roberta_rct500': [
        'allenai/dsp_roberta_base_dapt_biomed_tapt_chemprot_4169', 16, 2e-5, 0,
        False
    ],
    'biomed_roberta_chemprot':
    ['allenai/dsp_roberta_base_dapt_biomed_tapt_rct_500', 16, 2e-5, 0, False]
}

# ---------------------------------------------------------------------------
# Mapping from NER tag to ID
NER_TAG2ID = {'O': 0, 'B-RES': 1, 'I-RES': 2}

# Mapping from ID to NER tag
ID2NER_TAG = {v: k for k, v in NER_TAG2ID.items()}


# ---------------------------------------------------------------------------
def set_random_seed(seed):
    """
    Sets random seed for deterministic outcome of ML-trained models
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
class CustomHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """ Custom Argparse help formatter """
    def _get_help_string(self, action):
        """ Suppress defaults that are None """
        if action.default is None:
            return action.help
        return super()._get_help_string(action)

    def _format_action_invocation(self, action):
        """ Show metavar only once """
        if not action.option_strings:
            metavar, = self._metavar_formatter(action, action.dest)(1)
            return metavar
        parts = []
        if action.nargs == 0:
            parts.extend(action.option_strings)
        else:
            default = action.dest.upper()
            args_string = self._format_args(action, default)
            for option_string in action.option_strings:
                parts.append('%s' % option_string)
            parts[-1] += ' %s' % args_string
        return ', '.join(parts)


# ---------------------------------------------------------------------------
def strip_xml(text: str) -> str:
    """
    Strip XML tags from a string

    Parameters:
    `text`: String possibly containing XML tags

    Returns:
    String without XML tags
    """
    # If header tag between two adjacent strings, replace with a space
    pattern = re.compile(
        r'''(?<=[\w.?!]) # Header tag must be preceded by word
            (</?h[\d]>) # Header tag has letter h and number
            (?=[\w]) # Header tag must be followed by word''', re.X)
    text = re.sub(pattern, ' ', text)

    # Remove all other XML tags
    text = re.sub(r'<[\w/]+>', '', text)

    return text


# ---------------------------------------------------------------------------
def test_strip_xml() -> None:
    """ Test strip_xml() """

    assert strip_xml('<h4>Supplementary info</h4>') == 'Supplementary info'
    assert strip_xml('H<sub>2</sub>O<sub>2</sub>') == 'H2O2'
    assert strip_xml(
        'the <i>Bacillus pumilus</i> group.') == 'the Bacillus pumilus group.'

    # If there are not spaces around header tags, add them
    assert strip_xml(
        'MS/MS spectra.<h4>Availability') == 'MS/MS spectra. Availability'
    assert strip_xml('http://proteomics.ucsd.edu/Software.html<h4>Contact'
                     ) == 'http://proteomics.ucsd.edu/Software.html Contact'
    assert strip_xml(
        '<h4>Summary</h4>Neuropeptides') == 'Summary Neuropeptides'
    assert strip_xml('<h4>Wow!</h4>Go on') == 'Wow! Go on'


# ---------------------------------------------------------------------------
class Splits(NamedTuple):
    """
    Training, validation, and test dataframes

    `train`: Training data
    `val`: Validation data
    `test`: Test data
    """
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


# ---------------------------------------------------------------------------
def split_df(df: pd.DataFrame, rand_seed: bool, splits: List[float]) -> Splits:
    """
    Split manually curated data into train, validation and test sets

    `df`: Manually curated classification data
    `rand_seed`: Optionally use random seed
    `splits`: Proportions of data for [train, validation, test]

    Return:
    train, validation, test dataframes
    """

    seed = 241 if rand_seed else None

    _, val_split, test_split = splits
    val_test_split = val_split + test_split

    train, val_test = train_test_split(df,
                                       test_size=val_test_split,
                                       random_state=seed)
    val, test = train_test_split(val_test,
                                 test_size=test_split / val_test_split,
                                 random_state=seed)

    return Splits(train, val, test)


# ---------------------------------------------------------------------------
def test_random_split(unsplit_data: pd.DataFrame) -> None:
    """ Test that split_df() gives correct proportions """

    in_df = unsplit_data

    train, val, test = split_df(in_df, False, [0.5, 0.25, 0.25])

    assert len(train.index) == 4
    assert len(val.index) == 2
    assert len(test.index) == 2


# ---------------------------------------------------------------------------
def test_seeded_split(unsplit_data: pd.DataFrame) -> None:
    """ Test that split_df() behaves deterministically """

    in_df = unsplit_data

    train, val, test = split_df(in_df, True, [0.5, 0.25, 0.25])

    assert list(train['id'].values) == [321, 789, 741, 654]
    assert list(val['id'].values) == [987, 456]
    assert list(test['id'].values) == [852, 123]
