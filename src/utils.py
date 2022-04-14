"""
Purpose: Provide shared data structures and classes
Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import argparse

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
