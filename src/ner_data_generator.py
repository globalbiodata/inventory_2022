#!/usr/bin/env python3
"""
Purpose: Split curated NER data into training, validation, and testing sets
Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import argparse
import os
import string
import sys
from typing import List, NamedTuple, TextIO

import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import CustomHelpFormatter, Splits, strip_xml

# nltk.download('punkt')
# RND_SEED = 241


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """
    Command-line arguments

    `infile`: Input curated data filehandle
    `outdir`: Output directory
    `train`: Training data output file name
    `val`: Validation data output file name
    `spltis`: Train, val, test proportions
    `test`: Test data output file name
    `seed`: Random seed
    """
    infile: TextIO
    outdir: str
    train: str
    val: str
    test: str
    splits: List[float]
    seed: bool


# ---------------------------------------------------------------------------
def get_args() -> Args:
    """ Parse command-line arguments """

    parser = argparse.ArgumentParser(
        description='Split curated classification data',
        formatter_class=CustomHelpFormatter)

    parser.add_argument('infile',
                        metavar='FILE',
                        type=argparse.FileType('rt', encoding='ISO-8859-1'),
                        default='data/manual_ner_extraction.csv',
                        help='Manually classified input file')
    parser.add_argument('-o',
                        '--outdir',
                        metavar='',
                        type=str,
                        default='data/',
                        help='Output directory')
    parser.add_argument('-t',
                        '--train',
                        metavar='',
                        type=str,
                        default='train_ner.pkl',
                        help='Training data output file name')
    parser.add_argument('-v',
                        '--val',
                        metavar='',
                        type=str,
                        default='val_ner.pkl',
                        help='Validation data output file name')
    parser.add_argument('-s',
                        '--test',
                        metavar='',
                        type=str,
                        default='test_ner.pkl',
                        help='Test data output file name')
    parser.add_argument('--splits',
                        metavar='',
                        type=float,
                        nargs=3,
                        default=[0.7, 0.15, 0.15],
                        help='Proportions for train, val, test splits')
    parser.add_argument('-r',
                        '--seed',
                        action='store_true',
                        help='Set random seed')

    args = parser.parse_args()

    if not sum(args.splits) == 1.0:
        parser.error(f'--splits {args.splits} must sum to 1')

    return Args(args.infile, args.outdir, args.train, args.val, args.test,
                args.splits, args.seed)


# ---------------------------------------------------------------------------
def check_input(df: pd.DataFrame) -> None:
    """
    Check the input data columns

    `df`: Input dataframe
    """

    exp_cols = ['id', 'title', 'abstract', 'full_name', 'common_name']

    if not all(col in df.columns for col in exp_cols):
        sys.exit(
            f'ERROR: Input data does not have the expected columns: {exp_cols}'
        )


# --------------------------------------------------------------------------
def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Get only relevant data """

    return df[['id', 'title', 'abstract', 'full_name', 'common_name']]


# --------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Strip XML tags and deduplicate """

    df['title'] = df['title'].apply(strip_xml)
    df['abstract'] = df['abstract'].apply(strip_xml)

    df = df.drop_duplicates()

    return df


# ---------------------------------------------------------------------------
def get_offsets(text_sentences, resource_name, resource_type):
    """
    Matches a given resource_name (eg 'MEGALEX') of a given resource_type (eg RES) to a given list of sentences
    :param text_sentences: sentences to map the given resource_name to
    :param resource_name: resource_name to map
    :param resource_type: type of resource to map
    :return: list of words, word_indices, tags and sent_indices corresponding to the mapped sequence
    """
    word_indices = []
    tags = []
    words = []
    sent_indices = []

    for sent_idx, text in enumerate(text_sentences):
        text_tokens = text.split()
        text_tokens_stripped = np.array(
            [x.strip(string.punctuation).lower() for x in text_tokens])
        text_token_idx2tag = {}
        if resource_name == resource_name:

            resource_name_tokens = [x.lower() for x in resource_name.split()]
            resource_name_first_token = resource_name_tokens[0]

            match_indices = np.where(
                text_tokens_stripped == resource_name_first_token)[0]
            for match_idx in match_indices:
                found_match = True
                text_token_idx2tag[match_idx] = 'B-' + resource_type
                for i, resource_name_token in enumerate(
                        resource_name_tokens[1:]):
                    resource_name_token_in_range = (
                        (i + match_idx + 1) < (len(text_tokens_stripped)))
                    if (not resource_name_token_in_range) or (
                            resource_name_token_in_range and
                        (text_tokens_stripped[i + match_idx + 1] !=
                         resource_name_token)):
                        found_match = False
                    else:
                        text_token_idx2tag[match_idx + i +
                                           1] = 'I-' + resource_type
                if not found_match:
                    for i, resource_name_token in enumerate(
                            resource_name_tokens):
                        text_token_idx2tag[match_idx + i] = 'O'

        for token_idx, token in enumerate(text_tokens):
            words.append(token)
            word_indices.append(token_idx)
            tags.append(text_token_idx2tag[token_idx] if token_idx in
                        text_token_idx2tag else 'O')
            sent_indices.append(sent_idx)
    return words, word_indices, tags, sent_indices


# ---------------------------------------------------------------------------
def reconcile_tags(tags_arr1, tags_arr2):
    """
    Reconciles different set of tags in arrays corresponding to the same sequence of words.
    Eg: [B-RES, I-RES, O] and [O, O, O]
    Each word will get the more specific tag found in either of the arrays
    :param tags_arr1: array containing first list of tags
    :param tags_arr2: array containing second list of tags
    :return final_tags: array containing reconciled tags
    """
    final_tags = []
    for tag1, tag2 in zip(tags_arr1, tags_arr2):
        if tag1 == tag2:
            final_tags.append(tag1)
        elif tag1 != 'O':
            final_tags.append(tag1)
        else:
            final_tags.append(tag2)
    return final_tags


# ---------------------------------------------------------------------------
def BIO_scheme_transform(df):
    """
    Matches B-RES and I-RES tags according to the BIO-scheme for the mentions found under the 'name' and 'full_name' fields. 
    Matches on both the 'title' and 'abstract' fields, parsed to remove XML tags
    :param df: the given df. Must contain    the following fields: [id, title, abstract_parsed_xml, name, full_name]
    :return df: df containing sentences where mentions under 'name' and 'full_name' fields are being matched
    """
    pmids = df['id'].values
    titles = df['title'].values
    abstracts = df['abstract'].values
    names = df['common_name'].values
    full_names = df['full_name'].values

    all_words = []
    all_word_indices = []
    all_tags = []
    all_pmids = []
    all_sent_indices = []
    last_pmid = -1
    last_final_tags = []
    last_words = []
    last_word_indices = []
    last_sent_indices = []
    for pmid, title, abstract, resource_name, full_name in zip(
            pmids, titles, abstracts, names, full_names):
        title_abstract_sentences = nltk.sent_tokenize(
            title) + nltk.sent_tokenize(abstract)
        words, word_indices, tags, sentences_indices = get_offsets(
            title_abstract_sentences, resource_name, 'RES')
        _, _, full_names_tags, full_names_sentences_indices = get_offsets(
            title_abstract_sentences, full_name, 'RES')

        final_tags = reconcile_tags(tags, full_names_tags)
        if pmid == last_pmid:
            final_tags = reconcile_tags(final_tags, last_final_tags)
        # seeing a new pmid -> append last information unless last_pmid is -1
        elif last_pmid != -1:
            all_words.extend(last_words)
            all_word_indices.extend(last_word_indices)
            all_tags.extend(last_final_tags)
            all_pmids.extend([last_pmid] * len(last_words))
            all_sent_indices.extend(last_sent_indices)

        last_pmid = pmid
        last_final_tags = final_tags
        last_words = words
        last_word_indices = word_indices
        last_sent_indices = sentences_indices

    all_words.extend(last_words)
    all_word_indices.extend(last_word_indices)
    all_tags.extend(last_final_tags)
    all_pmids.extend([last_pmid] * len(last_words))
    all_sent_indices.extend(last_sent_indices)
    df = pd.DataFrame({
        'pmid': all_pmids,
        'sent_idx': all_sent_indices,
        'word': all_words,
        'word_idx': all_word_indices,
        'tag': all_tags
    })
    return df


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

    ids = df['pmid'].unique()
    train_ids, val_test_ids = train_test_split(ids,
                                               test_size=val_test_split,
                                               random_state=seed)
    val_ids, test_ids = train_test_split(val_test_ids,
                                         test_size=test_split / val_test_split,
                                         random_state=seed)

    train = df[df['pmid'].isin(train_ids)]
    val = df[df['pmid'].isin(val_ids)]
    test = df[df['pmid'].isin(test_ids)]

    return Splits(train, val, test)


# ---------------------------------------------------------------------------
def process_df(df, filename):
    """
    Saves a df as a pickle file under a given filename
    :param filename: Output filename
    """
    df_grouped = df.groupby(['pmid', 'sent_idx']).agg(list).reset_index()
    df_grouped = df_grouped.rename(columns={
        'word': 'words',
        'tag': 'ner_tags'
    })
    df_grouped.to_pickle(filename)


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()
    out_dir = args.outdir

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    df = pd.read_csv(args.infile)

    check_input(df)

    df = filter_data(df)

    df = clean_data(df)

    # df = df[~df['name'].isna()]
    ner_df = BIO_scheme_transform(df)

    # np.random.seed(RND_SEED)
    # sent_ids = ner_df['pmid'].unique()

    train_df, val_df, test_df = split_df(ner_df, args.seed, args.splits)

    train_out, val_out, test_out = map(lambda f: os.path.join(out_dir, f),
                                       [args.train, args.val, args.test])

    assert (len(set(train_df['pmid']).intersection(set(val_df['pmid']))) == 0)
    assert (len(set(train_df['pmid']).intersection(set(test_df['pmid']))) == 0)
    assert (len(set(val_df['pmid']).intersection(set(test_df['pmid']))) == 0)

    process_df(train_df, train_out)
    process_df(val_df, val_out)
    process_df(test_df, test_out)

    print(f'Done. Wrote 3 files to {out_dir}.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
