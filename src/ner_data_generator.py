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
import pandas as pd
from pandas._testing.asserters import assert_series_equal
from pandas.testing import assert_frame_equal
from sklearn.model_selection import train_test_split

from utils import CustomHelpFormatter, Splits, concat_title_abstract, strip_xml

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
class LabeledSentence(NamedTuple):
    """
    Sentence labeled with BIO scheme

    `words`: List of words in sentence
    `indices`: Word indices
    `tags`: BIO tag per word
    """
    words: List[str]
    indices: List[int]
    tags: List[str]


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
    df = df.fillna('')

    df = df.drop_duplicates()

    return df


# ---------------------------------------------------------------------------
def restructure_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a row for each word in article title and abstract
    Add sentence and word index columns

    `df`: Dataframe for single article with id, title_abstract, common_name
        and full_name columns

    Return:
    Dataframe with one row per token and with word and sentence indices
    """

    out_df = df.drop(['common_name', 'full_name'], axis='columns')

    out_df = df.set_index(['id'], append=True)
    out_df = out_df.title_abstract.map(nltk.sent_tokenize).apply(pd.Series)
    out_df = out_df.stack()
    out_df = out_df.reset_index(level=2, drop=True)
    out_df = out_df.reset_index(name='sentence')
    out_df = out_df.set_index(['id'], append=True)
    out_df = out_df.sentence.str.split(expand=True)
    out_df = out_df.stack()
    out_df = out_df.reset_index(name='word')
    out_df = out_df.rename(columns={
        'level_0': 'sent_idx',
        'level_2': 'word_idx',
        'id': 'pmid'
    })

    out_df = out_df[['pmid', 'sent_idx', 'word_idx', 'word']]

    return out_df


# ---------------------------------------------------------------------------
def test_restructure_df() -> None:
    """ Test restructure_df() """

    in_df = pd.DataFrame(
        [[
            456, 'The Auditory English Lexicon Project: A multi. (AELP) is a.',
            'Auditory English Lexicon Project', 'AELP'
        ]],
        columns=['id', 'title_abstract', 'full_name', 'common_name'])

    out_df = pd.DataFrame(
        [[456, 0, 0, 'The'], [456, 0, 1, 'Auditory'], [456, 0, 2, 'English'],
         [456, 0, 3, 'Lexicon'], [456, 0, 4, 'Project:'], [456, 0, 5, 'A'],
         [456, 0, 6, 'multi.'], [456, 1, 0, '(AELP)'], [456, 1, 1, 'is'],
         [456, 1, 2, 'a.']],
        columns=['pmid', 'sent_idx', 'word_idx', 'word'])

    assert_frame_equal(restructure_df(in_df), out_df)


# ---------------------------------------------------------------------------
def assign_tags(words: pd.Series, full_name: str,
                common_name: str) -> pd.Series:
    """
    Assign BIO tags to tokens in sequence

    `words`: Series of tokens stripped of punctuation
    `full_name`: Resource long name
    `common_name`: Resource common name

    Return:
    Series of tags (`O`, `B-COM`, `I-COM`, `B-FUL`, or `I-FUL`) corresponding
    to tokens in sequence
    """

    full_name = '' if common_name in full_name else full_name
    full_name_split = full_name.split(' ')
    common_name_split = common_name.split(' ')

    full_name_len = len(full_name_split)
    common_name_len = len(common_name_split)

    seq_len = len(words)

    tags = pd.Series(['O'] * seq_len)
    for i in range(len(words)):
        if i + common_name_len <= seq_len:
            if all(words[i:i + common_name_len] == common_name_split):
                tags[i] = 'B-COM'
                tags[i + 1:i + common_name_len] = 'I-COM'
        if i + full_name_len <= seq_len:
            if all(words[i:i + full_name_len] == full_name_split):
                tags[i] = 'B-FUL'
                tags[i + 1:i + full_name_len] = 'I-FUL'

    return tags


# ---------------------------------------------------------------------------
def test_assign_tags() -> None:
    """ Test assign_tags() """

    words = pd.Series(
        'The database of peptide ligand DPL is a database'.split(' '))
    full_name = 'database of peptide ligand'
    common_name = 'DPL'

    tags = pd.Series(
        ['O', 'B-FUL', 'I-FUL', 'I-FUL', 'I-FUL', 'B-COM', 'O', 'O', 'O'])

    assert_series_equal(assign_tags(words, full_name, common_name), tags)


# ---------------------------------------------------------------------------
def tag_article_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply BIO tagging to single article dataframe

    `df`: Dataframe for single article with id, title_abstract, common_name
        and full_name columns

    Return:
    Dataframe with one row per token and with word and sentence indices and
    BIO tags
    """

    full_name = df['full_name'].iloc[0]
    common_name = df['common_name'].iloc[0]
    out_df = restructure_df(df)

    words = out_df['word'].str.strip(string.punctuation)

    out_df['tag'] = assign_tags(words, full_name, common_name)

    out_df = out_df[['pmid', 'sent_idx', 'word_idx', 'tag', 'word']]

    return out_df


# ---------------------------------------------------------------------------
def test_tag_article_tokens() -> None:
    """ Test tag_article_tokens() """

    in_df = pd.DataFrame(
        [[
            456, 'The Auditory English Lexicon Project: A multi. (AELP) is a.',
            'Auditory English Lexicon Project', 'AELP'
        ]],
        columns=['id', 'title_abstract', 'full_name', 'common_name'])

    out_df = pd.DataFrame(
        [[456, 0, 0, 'O', 'The'], [456, 0, 1, 'B-FUL', 'Auditory'],
         [456, 0, 2, 'I-FUL', 'English'], [456, 0, 3, 'I-FUL', 'Lexicon'],
         [456, 0, 4, 'I-FUL', 'Project:'], [456, 0, 5, 'O', 'A'],
         [456, 0, 6, 'O', 'multi.'], [456, 1, 0, 'B-COM', '(AELP)'],
         [456, 1, 1, 'O', 'is'], [456, 1, 2, 'O', 'a.']],
        columns=['pmid', 'sent_idx', 'word_idx', 'tag', 'word'])

    assert_frame_equal(tag_article_tokens(in_df), out_df)

    # Partial matches to named entities should not be tagged
    in_df = pd.DataFrame(
        [[
            456, 'The database of peptide ligand (DPL) is a database.',
            'database of peptide ligand', 'DPL'
        ]],
        columns=['id', 'title_abstract', 'full_name', 'common_name'])

    out_df = pd.DataFrame(
        [[456, 0, 0, 'O', 'The'], [456, 0, 1, 'B-FUL', 'database'],
         [456, 0, 2, 'I-FUL', 'of'], [456, 0, 3, 'I-FUL', 'peptide'],
         [456, 0, 4, 'I-FUL', 'ligand'], [456, 0, 5, 'B-COM', '(DPL)'],
         [456, 0, 6, 'O', 'is'], [456, 0, 7, 'O', 'a'],
         [456, 0, 8, 'O', 'database.']],
        columns=['pmid', 'sent_idx', 'word_idx', 'tag', 'word'])

    assert_frame_equal(tag_article_tokens(in_df), out_df)


# ---------------------------------------------------------------------------
def BIO_scheme_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform BIO tagging for all articles in dataset

    `df`: Dataframe with one row per article including extracted resource
    common name and full name

    Return: Dataframe with one row per token per article including indices
    and BIO tags
    """

    df = concat_title_abstract(df)

    # df = resolve_nested_tags(df)

    out_df = pd.DataFrame()
    for _, article_df in df.groupby('id'):
        tagged_df = tag_article_tokens(article_df)
        out_df = pd.concat([out_df, tagged_df])

    out_df = out_df.reset_index(drop=True)

    return out_df


# ---------------------------------------------------------------------------
def test_BIO_scheme_transform() -> None:
    """ Test BIO_scheme_transform() """

    in_df = pd.DataFrame(
        [[
            123, 'MEGALEX: A megastudy.', 'New database (MEGALEX) of.',
            'MEGALEX', 'MEGALEX'
        ],
         [
             456, 'The Auditory English Lexicon Project: A multi.',
             '(AELP) is a.', 'Auditory English Lexicon Project', 'AELP'
         ]],
        columns=['id', 'title', 'abstract', 'full_name', 'common_name'])

    out_df = pd.DataFrame(
        [[123, 0, 0, 'B-COM', 'MEGALEX:'], [123, 0, 1, 'O', 'A'],
         [123, 0, 2, 'O', 'megastudy.'], [123, 1, 0, 'O', 'New'],
         [123, 1, 1, 'O', 'database'], [123, 1, 2, 'B-COM', '(MEGALEX)'],
         [123, 1, 3, 'O', 'of.'], [456, 0, 0, 'O', 'The'],
         [456, 0, 1, 'B-FUL', 'Auditory'], [456, 0, 2, 'I-FUL', 'English'],
         [456, 0, 3, 'I-FUL', 'Lexicon'], [456, 0, 4, 'I-FUL', 'Project:'],
         [456, 0, 5, 'O', 'A'], [456, 0, 6, 'O', 'multi.'],
         [456, 1, 0, 'B-COM', '(AELP)'], [456, 1, 1, 'O', 'is'],
         [456, 1, 2, 'O', 'a.']],
        columns=['pmid', 'sent_idx', 'word_idx', 'tag', 'word'])

    assert_frame_equal(BIO_scheme_transform(in_df), out_df, check_dtype=False)

    # Tokens cannot have multiple tags, so if common_name is in full_name
    # Do not tag full_name
    in_df = pd.DataFrame(
        [[
            456, 'The Ensembl project.', 'Is a thing.', 'Ensembl project',
            'Ensembl'
        ]],
        columns=['id', 'title', 'abstract', 'full_name', 'common_name'])

    out_df = pd.DataFrame(
        [[456, 0, 0, 'O', 'The'], [456, 0, 1, 'B-COM', 'Ensembl'],
         [456, 0, 2, 'O', 'project.'], [456, 1, 0, 'O', 'Is'],
         [456, 1, 1, 'O', 'a'], [456, 1, 2, 'O', 'thing.']],
        columns=['pmid', 'sent_idx', 'word_idx', 'tag', 'word'])

    assert_frame_equal(BIO_scheme_transform(in_df), out_df)


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
def process_df(df: pd.DataFrame, filename: str) -> None:
    """
    Save dataframe as pickle

    `df`: Dataframe to be pickled
    `filename`: Output filename
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

    df = pd.read_csv(args.infile, )

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

    process_df(train_df, train_out)
    process_df(val_df, val_out)
    process_df(test_df, test_out)

    print(f'Done. Wrote 3 files to {out_dir}.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
