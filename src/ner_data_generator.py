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
from numpy.core.numeric import NaN
from pandas._testing.asserters import assert_series_equal
from pandas.testing import assert_frame_equal

from utils import (CustomHelpFormatter, concat_title_abstract, split_df,
                   strip_xml)

# nltk.download('punkt')
# RND_SEED = 241


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """
    Command-line arguments

    `infile`: Input curated data filehandle
    `outdir`: Output directory
    `splits`: Train, val, test proportions
    `seed`: Random seed
    """
    infile: TextIO
    outdir: str
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

    return Args(args.infile, args.outdir, args.splits, args.seed)


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
    """
    Filter input data for completeness and relevant columns

    `df`: Input data dataframe

    Return: Filtered dataframe
    """

    # Filter out rows that are missing both full_name and common_name
    df = df.dropna(subset=['full_name', 'common_name'], how='all')

    df = df.reset_index(drop=True)

    return df[['id', 'title', 'abstract', 'full_name', 'common_name']]


# --------------------------------------------------------------------------
def test_filter_data() -> None:
    """ Test filter_data() """

    in_df = pd.DataFrame(
        [['123', 'A title', 'An abstract.', NaN, NaN, '', ''],
         ['456', 'A title', 'An abstract.', 'full_name', NaN, '', ''],
         ['789', 'A title', 'An abstract.', NaN, 'common_name', '', '']],
        columns=[
            'id', 'title', 'abstract', 'full_name', 'common_name', 'url',
            'short_description'
        ])

    out_df = pd.DataFrame(
        [['456', 'A title', 'An abstract.', 'full_name', NaN],
         ['789', 'A title', 'An abstract.', NaN, 'common_name']],
        columns=['id', 'title', 'abstract', 'full_name', 'common_name'])

    assert_frame_equal(filter_data(in_df), out_df)


# --------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip XML tags, replace NAs, deduplicate

    `df`: Input data dataframe

    Return: Cleaned dataframe
    """

    df['title'] = df['title'].apply(strip_xml)
    df['abstract'] = df['abstract'].apply(strip_xml)
    df = df.fillna('')

    df = df.drop_duplicates()

    return df


# --------------------------------------------------------------------------
def test_clean_data() -> None:
    """ Test clean_data() """

    in_df = pd.DataFrame(
        [['123', 'A <i>title</i>', 'An <i>abstract</i>.', 'full_name', NaN],
         ['456', 'A dup title', 'A dup abstract.', 'full_name', 'common_name'],
         ['456', 'A dup title', 'A dup abstract.', 'full_name', 'common_name']
         ],
        columns=['id', 'title', 'abstract', 'full_name', 'common_name'])

    out_df = pd.DataFrame(
        [['123', 'A title', 'An abstract.', 'full_name', ''],
         ['456', 'A dup title', 'A dup abstract.', 'full_name', 'common_name']
         ],
        columns=['id', 'title', 'abstract', 'full_name', 'common_name'])

    assert_frame_equal(clean_data(in_df), out_df)


# --------------------------------------------------------------------------
def combine_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine rows of same id into single row

    `df`: Dataframe with potentially multiple rows per id

    Return: Dataframe with single row per id
    """

    out_df = pd.DataFrame(
        columns=['id', 'title', 'abstract', 'full_name', 'common_name'])
    for article_id, article in df.groupby('id'):
        title = article.title.values[0]
        abstract = article.abstract.values[0]
        full_names = sorted(list(set(article.full_name.values)))
        common_names = sorted(list(set(article.common_name.values)))

        row = pd.DataFrame(
            [[article_id, title, abstract, full_names, common_names]],
            columns=['id', 'title', 'abstract', 'full_name', 'common_name'])
        out_df = pd.concat([out_df, row])

    out_df = out_df.reset_index(drop=True)

    return out_df


# --------------------------------------------------------------------------
def test_combine_rows() -> None:
    """ Test combine_rows() """

    in_df = pd.DataFrame(
        [['123', 'MEGALEX', 'An abstract', '', 'MEGALEX'],
         ['456', 'CircR2Cancer', 'circR2Cancer', 'foo', 'CircR2Cancer'],
         ['456', 'CircR2Cancer', 'circR2Cancer', 'foo', 'circR2Cancer']],
        columns=['id', 'title', 'abstract', 'full_name', 'common_name'])

    out_df = pd.DataFrame(
        [['123', 'MEGALEX', 'An abstract', [''], ['MEGALEX']],
         [
             '456', 'CircR2Cancer', 'circR2Cancer', ['foo'],
             ['CircR2Cancer', 'circR2Cancer']
         ]],
        columns=['id', 'title', 'abstract', 'full_name', 'common_name'])

    assert_frame_equal(combine_rows(in_df), out_df)


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
            ['Auditory English Lexicon Project'], ['AELP']
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
def assign_tags(words: pd.Series, full_names: List[str],
                common_names: List[str]) -> pd.Series:
    """
    Assign BIO tags to tokens in sequence

    `words`: Series of tokens stripped of punctuation
    `full_name`: Resource long name
    `common_name`: Resource common name

    Return:
    Series of tags (`O`, `B-COM`, `I-COM`, `B-FUL`, or `I-FUL`) corresponding
    to tokens in sequence
    """

    seq_len = len(words)

    tags = pd.Series(['O'] * seq_len)
    for i in range(seq_len):
        for common_name in common_names:
            common_name_split = common_name.split(' ')
            common_name_len = len(common_name_split)
            if i + common_name_len <= seq_len:
                if all(words[i:i + common_name_len] == common_name_split):
                    tags[i] = 'B-COM'
                    tags[i + 1:i + common_name_len] = 'I-COM'
        for full_name in full_names:
            full_name = '' if any(name in full_name
                                  for name in common_names) else full_name
            full_name_split = full_name.split(' ')
            full_name_len = len(full_name_split)
            if i + full_name_len <= seq_len:
                if all(words[i:i + full_name_len] == full_name_split):
                    tags[i] = 'B-FUL'
                    tags[i + 1:i + full_name_len] = 'I-FUL'

    return tags


# ---------------------------------------------------------------------------
def test_assign_tags() -> None:
    """ Test assign_tags() """

    # Partial matches to named entities should not be tagged
    words = pd.Series(
        'The database of peptide ligand DPL is a database'.split(' '))
    full_names = ['database of peptide ligand']
    common_names = ['DPL']
    tags = pd.Series(
        ['O', 'B-FUL', 'I-FUL', 'I-FUL', 'I-FUL', 'B-COM', 'O', 'O', 'O'])

    assert_series_equal(assign_tags(words, full_names, common_names), tags)

    # Able to tag multiple entities per category
    words = pd.Series('CircR2DNA is a database circR2DNA'.split(' '))
    full_names = ['']
    common_names = ['circR2DNA', 'CircR2DNA']
    tags = pd.Series(['B-COM', 'O', 'O', 'O', 'B-COM'])

    assert_series_equal(assign_tags(words, full_names, common_names), tags)

    # Tokens cannot have multiple tags, so if common_name is in full_name
    # Do not tag full_name
    words = pd.Series('The Ensembl project is a thing'.split(' '))
    full_names = ['Ensembl project']
    common_names = ['Ensembl']
    tags = pd.Series(['O', 'B-COM', 'O', 'O', 'O', 'O'])

    assert_series_equal(assign_tags(words, full_names, common_names), tags)


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
            ['Auditory English Lexicon Project'], ['AELP']
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
            ['MEGALEX'], ['MEGALEX']
        ],
         [
             456, 'The Auditory English Lexicon Project: A multi.',
             '(AELP) is a.', ['Auditory English Lexicon Project'], ['AELP']
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


# ---------------------------------------------------------------------------
def group_tagged_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group dataframe by pmid and sentence index

    `df`: Dataframe to be grouped
    """
    df_grouped = df.groupby(['pmid', 'sent_idx']).agg(list).reset_index()
    df_grouped = df_grouped.rename(columns={
        'word': 'words',
        'tag': 'ner_tags'
    })

    return df_grouped


# ---------------------------------------------------------------------------
def test_group_tagged_df() -> None:
    """ Test group_tagged_df() """

    in_df = pd.DataFrame(
        [[123, 0, 0, 'B-COM', 'MEGALEX:'], [123, 0, 1, 'O', 'A'],
         [123, 0, 2, 'O', 'megastudy.'], [123, 1, 0, 'O', 'New'],
         [123, 1, 1, 'O', 'database'], [123, 1, 2, 'B-COM', '(MEGALEX)'],
         [123, 1, 3, 'O', 'of.'], [456, 0, 0, 'O', 'The'],
         [456, 0, 1, 'B-FUL', 'Auditory'], [456, 0, 2, 'I-FUL', 'English'],
         [456, 0, 3, 'I-FUL', 'Lexicon'], [456, 0, 4, 'I-FUL', 'Project:'],
         [456, 0, 5, 'O', 'A'], [456, 0, 6, 'O', 'multi.'],
         [456, 1, 0, 'B-COM', '(AELP)'], [456, 1, 1, 'O', 'is'],
         [456, 1, 2, 'O', 'a.']],
        columns=['pmid', 'sent_idx', 'word_idx', 'ner_tags', 'words'])

    out_df = pd.DataFrame(
        [[
            123, 0, [0, 1, 2], ['B-COM', 'O', 'O'],
            ['MEGALEX:', 'A', 'megastudy.']
        ],
         [
             123, 1, [0, 1, 2, 3], ['O', 'O', 'B-COM', 'O'],
             ['New', 'database', '(MEGALEX)', 'of.']
         ],
         [
             456, 0, [0, 1, 2, 3, 4, 5, 6],
             ['O', 'B-FUL', 'I-FUL', 'I-FUL', 'I-FUL', 'O', 'O'],
             [
                 'The', 'Auditory', 'English', 'Lexicon', 'Project:', 'A',
                 'multi.'
             ]
         ], [456, 1, [0, 1, 2], ['B-COM', 'O', 'O'], ['(AELP)', 'is', 'a.']]],
        columns=['pmid', 'sent_idx', 'word_idx', 'ner_tags', 'words'])

    assert_frame_equal(group_tagged_df(in_df), out_df, check_dtype=False)


# ---------------------------------------------------------------------------
def save_df(df: pd.DataFrame, filename: str) -> None:
    """
    Save dataframe to pickle

    `df`: Dataframe to be pickled
    `filename`: Output filename
    """

    df.to_pickle(filename)


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()
    out_dir = args.outdir

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    df = pd.read_csv(args.infile, )

    check_input(df)

    df = combine_rows(clean_data(filter_data(df)))

    raw_train, raw_val, raw_test = split_df(df, args.seed, args.splits)

    train_df, val_df, test_df = map(lambda df: BIO_scheme_transform(df),
                                    [raw_train, raw_val, raw_test])

    raw_train_out, raw_val_out, raw_test_out = map(
        lambda f: os.path.join(out_dir, f),
        ['train_ner.csv', 'val_ner.csv', 'test_ner.csv'])

    raw_train.to_csv(raw_train_out, index=False)
    raw_val.to_csv(raw_val_out, index=False)
    raw_test.to_csv(raw_test_out, index=False)

    train_out, val_out, test_out = map(
        lambda f: os.path.join(out_dir, f),
        ['train_ner.pkl', 'val_ner.pkl', 'test_ner.pkl'])

    save_df(group_tagged_df(train_df), train_out)
    save_df(group_tagged_df(val_df), val_out)
    save_df(group_tagged_df(test_df), test_out)

    print(f'Done. Wrote 3 files to {out_dir}.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
