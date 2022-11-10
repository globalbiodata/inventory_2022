#!/usr/bin/env python3
"""
Purpose: Process data that has been manually reviewed after flagging
Authors: Kenneth Schackart
"""

import argparse
import itertools
import os
import re
import sys
from typing import List, NamedTuple, TextIO, Tuple

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from inventory_utils.custom_classes import CustomHelpFormatter
from inventory_utils.wrangling import join_commas
from process_names import wrangle_names

pd.options.mode.chained_assignment = None


# ---------------------------------------------------------------------------
class Args(NamedTuple):
    """ Command-line arguments """
    file: TextIO
    out_dir: str


# ---------------------------------------------------------------------------
def get_args() -> Args:
    """ Parse command-line arguments """

    desc = 'Process data that has been manually reviewed after flagging'
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=CustomHelpFormatter)

    parser.add_argument('file',
                        metavar='FILE',
                        type=argparse.FileType('rt', encoding='ISO-8859-1'),
                        help='CSV file of articles')
    parser.add_argument('-o',
                        '--out-dir',
                        metavar='DIR',
                        type=str,
                        default='out/',
                        help='Output directory')

    args = parser.parse_args()

    return Args(args.file, args.out_dir)


# ---------------------------------------------------------------------------
@pytest.fixture(name='raw_data')
def fixture_raw_data() -> pd.DataFrame:
    """ DataFrame representative of the input data """

    columns = [
        'ID', 'text', 'extracted_url', 'best_common', 'best_common_prob',
        'best_full', 'best_full_prob', 'best_name', 'best_name_prob',
        'article_count', 'duplicate_urls', 'duplicate_names', 'low_prob',
        'review_low_prob', 'review_dup_urls', 'review_dup_names',
        'review_notes_low_prob', 'review_notes_dup_urls',
        'review_notes_dup_names', 'publication_date'
    ]

    df = pd.DataFrame(
        [
            [  # Nothing to be done
                '1', 'text1', 'url1', '', '', '', '', 'name1', '1.0', '1', '',
                '', '', '', '', '', '', '', '', '1/1/2011'
            ],
            [  # Flagged low name probability, marked keep
                '2', 'text2', 'url2', '', '', '', '', 'name2', '0.85', '1', '',
                '', 'low_prob_best_name', 'do not remove', '', '', '', '', '',
                '1/2/2011'
            ],
            [  # Flagged low name probability, marked remove
                '3', 'text3', 'url3', '', '', '', '', 'name3', '0.85', '1', '',
                '', 'low_prob_best_name', 'remove', '', '', '', '', '',
                '1/3/2011'
            ],
            [  # Same URL as 5, marked do not merge
                '4', 'text4', 'url4', '', '', '', '', 'name4', '0.96', '1',
                '5.0', '', '', '', 'do not merge', '', '', '', '', '1/4/2011'
            ],
            [  # Same URL as 4, marked do not merge
                '5', 'text5', 'url4', '', '', '', '', 'name5', '0.97', '1',
                '4.0', '', '', '', 'do not merge', '', '', '', '', '1/5/2011'
            ],
            [  # Same URL as 7, marked merge
                '6', 'text6', 'url6', 'name6', '0.96', '', '', 'name6', '0.96',
                '1', '7.0', '', '', '', 'merge on record with best name prob',
                '', '', '', '', '1/6/2011'
            ],
            [  # Same URL as 6, marked merge
                '7', 'text7', 'url6', 'name7', '0.97', '', '', 'name7', '0.97',
                '1', '6.0', '', '', '', 'merge on record with best name prob',
                '', '', '', '', '1/7/2011'
            ],
            [  # Same name as 9, marked do not merge
                '8', 'text8', 'url8', 'name8', '0.99', '', '', 'name8', '0.99',
                '1', '', '9', '', '', '', 'do not merge', '', '', '',
                '1/8/2011'
            ],
            [  # Same name as 8, marked do not merge
                '9', 'text9', 'url9', 'name8', '0.99', '', '', 'name8', '0.99',
                '1', '', '8', '', '', '', 'do not merge', '', '', '',
                '1/9/2011'
            ],
            [  # Same name as 11, marked merge all "dup name" IDs
                '10', 'text10', 'url10', 'name10', '0.99', '', '', 'name10',
                '0.99', '1', '', '11', '', '', '', 'merge all "dup name" IDs',
                '', '', '', '1/10/2011'
            ],
            [  # Same name as 10, marked merge all "dup name" IDs
                '11', 'text11', 'url11', 'name10', '0.98', '', '', 'name10',
                '0.98', '1', '', '10', '', '', '', 'merge all "dup name" IDs',
                '', '', '', '1/11/2011'
            ],
            [  # Same name as 13 and 14, marked do not merge
                '12', 'text12', 'url12', 'name12', '0.98', '', '', 'name12',
                '0.98', '1', '', '13, 14', '', '', '', 'do not merge', '', '',
                '', '1/12/2011'
            ],
            [  # Same name as 12 and 14 marked merge only:
                '13', 'text13', 'url13', 'name12', '0.99', '', '', 'name12',
                '0.99', '1', '', '12, 14', '', '', '', 'merge only:', '', '',
                '13, 14', '1/13/2011'
            ],
            [  # Same name as 12 and 13, marked merge only:
                '14', 'text14', 'url14', 'name12', '0.97', '', '', 'name12',
                '0.98', '1', '', '12, 13', '', '', '', 'merge only:', '', '',
                '13, 14', '1/14/2011'
            ],
            [  # Same name as 17
                '15', 'text15', 'url15', 'name15', '0.91', '', '', 'name15',
                '0.91', '1', '', '17', 'low_prob_best_name', 'do not remove',
                '', 'merge only:', '', '', '15, 17', '1/12/2011'
            ],
            [  # Same URL as 17, same name as 18
                '16', 'text17', 'url16', 'name16', '0.96', '', '', 'name16',
                '0.96', '1', '17', '18', 'low_prob_best_name', 'do not remove',
                'merge on record with best name prob', 'do not merge', '', '',
                '13, 14', '1/13/2011'
            ],
            [  # Same URL as 16, same name as 15
                '17', 'text17', 'url16', 'name15', '0.99', '', '', 'name15',
                '0.99', '1', '16', '15', '', '',
                'merge on record with best name prob', 'merge only:', '', '',
                '15, 17', '1/14/2011'
            ],
            [  # Same name as 16
                '18', 'text18', 'url18', 'name16', '0.98', '', '', 'name16',
                '0.98', '1', '', '16', '', '', '', 'do not merge', '', '',
                '13, 14', '1/14/2011'
            ]
        ],
        columns=columns)

    return df


# ---------------------------------------------------------------------------
def ids_are_ok(string: str) -> bool:
    """
    Check that the value in a column which should only contain IDs or a list
    of IDs is valid

    Parameters:
    `string`: Input string

    Return: `True` or `False` stating if the value is OK
    """

    # Can't be empty string
    if not string:
        return False

    allowable = '0123456789., '

    return all(char in allowable for char in string)


# ---------------------------------------------------------------------------
def test_ids_are_ok() -> None:
    """ Test ids_are_ok() """

    assert ids_are_ok('123456')
    assert ids_are_ok('123456.0')
    assert ids_are_ok('123456.0, 789456.0')
    assert ids_are_ok('123456, 789456')

    assert not ids_are_ok('')
    assert not ids_are_ok('ACH alphabetic characters!')


# ---------------------------------------------------------------------------
def check_manual_columns(df: pd.DataFrame) -> str:
    """
    Check that the correct manual review columns are present.
    If any columns are missing, an error message is returned as a string.
    If all columns are present, an empty string is returned.

    Parameters:
    `df`: Input dataframe

    Return: Error message string
    """

    # These are hard-coded because if they change, the logic of the
    # program will need to be updated.
    manual_cols = [
        'review_low_prob', 'review_dup_urls', 'review_dup_names',
        'review_notes_dup_names'
    ]

    exit_message = ''

    for col in manual_cols:
        if col not in df.columns:
            exit_message += f'ERROR: Manual review column {col} is missing.\n'

    return exit_message


# ---------------------------------------------------------------------------
def test_check_manual_columns() -> None:
    """ Test check_manual_columns()"""

    in_df = pd.DataFrame([['foo', 'bar', 'baz', 'qux']],
                         columns=['not', 'the', 'right', 'columns'])

    exit_message = check_manual_columns(in_df)

    assert exit_message != ''
    assert exit_message.count('ERROR:') == 4  # All 4 columns are missing
    assert exit_message.count('\n') == 4

    in_df = pd.DataFrame([['foo', 'bar', 'baz', 'qux']],
                         columns=[
                             'review_low_prob', 'review_dup_urls',
                             'review_dup_names', 'review_notes_dup_names'
                         ])

    assert check_manual_columns(in_df) == ''


# ---------------------------------------------------------------------------
def check_for_responses(df: pd.DataFrame) -> str:
    """
    Check that all flagged rows have been addressed in their appropricate
    manual review columns.
    Error message will give the ID of rows that are flagged but not
    addressed. If all rows are okay, an empty string is returned.

    Parameters:
    `df`: Input dataframe

    Return: Error message string
    """

    flag_cols = ['duplicate_urls', 'duplicate_names', 'low_prob']
    response_cols = ['review_dup_urls', 'review_dup_names', 'review_low_prob']

    exit_message = ''
    for flag_col, response_col in zip(flag_cols, response_cols):
        unresponded = df[(df[flag_col] != '') & (df[response_col] == '')]
        for row in unresponded.itertuples():
            exit_message += ('ERROR: Missing response to flagged column '
                             f'"{flag_col}" for ID {row.ID}\n')

    return exit_message


# ---------------------------------------------------------------------------
def test_check_for_responses() -> None:
    """ Test check_for_responses() """

    columns = [
        'ID', 'duplicate_urls', 'duplicate_names', 'low_prob',
        'review_low_prob', 'review_dup_urls', 'review_dup_names'
    ]

    in_df = pd.DataFrame([['123', '456', '', '', '', '', ''],
                          ['456', '123', '789', '', '', '', ''],
                          ['789', '', '456', '', '', '', ''],
                          ['147', '', '', 'low_prob_best_name', '', '', '']],
                         columns=columns)

    exit_message = check_for_responses(in_df)

    assert exit_message != ''
    assert re.findall('"duplicate_urls" for ID 123', exit_message)
    assert re.findall('"duplicate_names" for ID 456', exit_message)
    assert re.findall('"duplicate_names" for ID 789', exit_message)
    assert re.findall('"low_prob" for ID 147', exit_message)

    in_df = pd.DataFrame(
        [['123', '456', '', '', '', 'do not merge', ''],
         ['456', '123', '789', '', '', 'do not merge', 'do not merge'],
         ['789', '', '456', '', '', '', 'do not merge'],
         ['147', '', '', 'low_prob_best_name', 'remove', '', '']],
        columns=columns)

    exit_message = check_for_responses(in_df)

    assert exit_message == ''


# ---------------------------------------------------------------------------
def check_manual_column_values(df: pd.DataFrame) -> str:
    """
    Check that each manual review columns only contains allowed values.
    If any columns have invalid values, an error message is returned.
    If all columns are okay, an empty string is returned.

    Parameters:
    `df`: Input dataframe

    Return: Error message string
    """

    allowed_values = {
        'review_low_prob':
        set(['', 'remove', 'do not remove']),
        'review_dup_urls':
        set([
            '', 'merge on record with best name prob', 'do not merge',
            'conflicting record(s) to be removed'
        ]),
        'review_dup_names':
        set([
            '', 'merge all "dup name" IDs', 'do not merge', 'merge only:',
            'conflicting record(s) to be removed'
        ])
    }

    exit_message = ''
    for col in allowed_values:
        col_values = set(df[col].unique())
        bad_values = col_values - allowed_values[col]

        if bad_values:
            exit_message += (f'ERROR: Column "{col}" contains invalid values: '
                             f'{bad_values}\n')

    return exit_message


# ---------------------------------------------------------------------------
def test_check_manual_column_values() -> None:
    """ Test check_manual_column_values()"""

    in_df = pd.DataFrame([['foo', 'bar', 'baz', '']],
                         columns=[
                             'review_low_prob', 'review_dup_urls',
                             'review_dup_names', 'review_notes_dup_names'
                         ])

    exit_message = check_manual_column_values(in_df)

    assert exit_message != ''
    assert exit_message.count('ERROR:') == 3  # All 3 columns have bad values
    assert exit_message.count('\n') == 3

    in_df = pd.DataFrame(
        [['remove', 'conflicting record(s) to be removed', '', ''],
         ['do not remove', 'merge on record with best name prob', '', ''],
         ['do not remove', '', 'merge only:', '123, 456']],
        columns=[
            'review_low_prob', 'review_dup_urls', 'review_dup_names',
            'review_notes_dup_names'
        ])

    assert check_manual_column_values(in_df) == ''


# ---------------------------------------------------------------------------
def check_note_column_values(df: pd.DataFrame) -> str:
    """
    Check that if a manual review column says "merge only:", the
    corresponding note column contains IDs only.
    If the notes column contains anything else, an error message is returned.
    If the notes column is good, an empty string is returned.

    Parameters:
    `df`: Input dataframe

    Return: Error message string
    """

    exit_message = ''
    for row in df.itertuples():
        if row.review_dup_names == 'merge only:':
            ids = row.review_notes_dup_names
            if not ids_are_ok(ids):
                exit_message += (f'ERROR: Invalid IDs for ID {row.ID} '
                                 'in column "review_notes_dup_names": '
                                 f'{ids}.\n')

    return exit_message


# ---------------------------------------------------------------------------
def test_check_note_column_values() -> None:
    """ Test check_note_column_values() """

    in_df = pd.DataFrame([['123', '', '', 'merge only:', ''],
                          ['456', '', '', 'merge only:', 'Bad']],
                         columns=[
                             'ID', 'review_low_prob', 'review_dup_urls',
                             'review_dup_names', 'review_notes_dup_names'
                         ])

    exit_message = check_note_column_values(in_df)

    assert exit_message != ''
    assert exit_message.count('ERROR:') == 2
    assert exit_message.count('\n') == 2
    assert re.findall('123', exit_message)
    assert re.findall('456', exit_message)

    in_df = pd.DataFrame([[
        '123', 'remove', 'conflicting record(s) to be removed', '', ''
    ], ['456', 'do not remove', 'merge on record with best name prob', '', ''],
                          ['789', '', '', 'merge only:', '123, 456']],
                         columns=[
                             'ID', 'review_low_prob', 'review_dup_urls',
                             'review_dup_names', 'review_notes_dup_names'
                         ])

    assert check_note_column_values(in_df) == ''


# ---------------------------------------------------------------------------
def check_data(df: pd.DataFrame) -> None:
    """
    Check that dataframe has the necessary columns for processing, and that
    only valid values are in the manually filled columns. Raise an excpetion
    on invalid input.

    Parameters:
    `df`: Input dataframe

    Return: `None`
    """

    exit_message = check_manual_columns(df)

    if exit_message:
        sys.exit(exit_message)

    exit_message = check_for_responses(df)
    exit_message += check_manual_column_values(df)
    # exit_message += check_note_column_values(df)

    if exit_message:
        sys.exit(exit_message)


# ---------------------------------------------------------------------------
def remove_decimals(in_string: str) -> str:
    """
    Remove all decimal points and trailing 0's from a string

    Parameters:
    `in_string`: Input string of IDs

    Return: String without decimals
    """

    return in_string.replace('.0', '')


# ---------------------------------------------------------------------------
def test_remove_decimals() -> None:
    """ Test remove_decimals() """

    assert remove_decimals('123456.0') == '123456'
    assert remove_decimals('123456.0, 456789.0') == '123456, 456789'
    assert remove_decimals('123456') == '123456'


# ---------------------------------------------------------------------------
def reformat_date(date: str) -> str:
    """
    Reformat date from M/D/YYYY format to YYYY-MM-DD so that they can be
    sorted as strings properly

    Parameters:
    `date`: Date in M/D/YYYY format

    Return: Date in YYYY-MM-DD format
    """

    if date == '':
        return date

    match = re.match(
        r'''
                     (?P<month>\d{1,2})/ # 1 or 2 digit month
                     (?P<day>\d{1,2})/   # 1 or 2 digit day
                     (?P<year>\d{4})     # 4 digit year
                     ''', date, re.X)

    if not match:
        sys.exit('ERROR: Dates must be in M/D/YYYY format')

    year = match['year']
    month = match['month'] if len(
        match['month']) == 2 else '0' + match['month']
    day = match['day'] if len(match['day']) == 2 else '0' + match['day']

    return year + '-' + month + '-' + day


# ---------------------------------------------------------------------------
def test_reformat_date() -> None:
    """ Test reformat_date() """

    assert reformat_date('') == ''
    assert reformat_date('1/1/2011') == '2011-01-01'
    assert reformat_date('10/31/2012') == '2012-10-31'


# ---------------------------------------------------------------------------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataframe before further processing. Remove decimal points from
    columns with IDs

    Parameters:
    `df`: Input dataframe

    Return: Cleaned dataframe
    """

    df['ID'] = df['ID'].map(remove_decimals)
    df['duplicate_urls'] = df['duplicate_urls'].map(remove_decimals)
    df['duplicate_names'] = df['duplicate_names'].map(remove_decimals)
    df['review_notes_dup_names'] = df['review_notes_dup_names'].map(
        remove_decimals)
    df['publication_date'] = df['publication_date'].map(reformat_date)
    df.drop('text', axis='columns', inplace=True)

    return df


# ---------------------------------------------------------------------------
def test_clean_df() -> None:
    """ Test clean_df() """

    in_df = pd.DataFrame(
        [['123.0', 'text1', '456.0', '789.0', '147.0', '11/18/2021']],
        columns=[
            'ID', 'text', 'duplicate_urls', 'duplicate_names',
            'review_notes_dup_names', 'publication_date'
        ])

    out_df = pd.DataFrame([['123', '456', '789', '147', '2021-11-18']],
                          columns=[
                              'ID', 'duplicate_urls', 'duplicate_names',
                              'review_notes_dup_names', 'publication_date'
                          ])

    assert_frame_equal(clean_df(in_df), out_df)


# ---------------------------------------------------------------------------
def drop_low_probs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that have been marked for removal for having incorrect
    names that had been flagged due to low predictoin probability.

    Parameters:
    `df`: Input dataframe

    Return: Dataframe with rows removed
    """

    df = df.copy()

    df = df[df['review_low_prob'] != 'remove']
    df = df.drop(['low_prob', 'review_low_prob', 'review_notes_low_prob'],
                 axis='columns')

    return df


# ---------------------------------------------------------------------------
def test_drop_low_probs(raw_data: pd.DataFrame) -> None:
    """ Test drop_low_probs() """

    out_df = drop_low_probs(raw_data)
    remaining_ids = out_df['ID'].values

    assert '3' not in remaining_ids
    assert '2' in remaining_ids

    assert all(
        col not in out_df.columns
        for col in ['low_prob', 'review_low_prob', 'review_notes_low_prob'])


# ---------------------------------------------------------------------------
def check_instructions(id_col: pd.Series,
                       review_col: pd.Series) -> Tuple[List[str], str]:
    """
    Check that instructions for given rows are consistent.
    If not, return a string describing the problem.

    Parameters:
    `id_col`: Column of problematic IDs
    `review_col`: Column of conflicting instructions

    Return: tuple(Unique set of instrucions, error string or empty string)
    """

    exit_message = ''

    unpacked_instructions = [elem.split(', ') for elem in review_col]
    review_col_vals = pd.Series(itertools.chain(*unpacked_instructions))

    instructions = review_col_vals.unique()
    if len(instructions) != 1:
        conflicts = [
            f'{id_i}: "{msg_i}"'
            for id_i, msg_i in zip(id_col.values, review_col.values)
        ]
        exit_message = (
            f'ERROR: Conflicting instructions in column {review_col.name}:\n' +
            '\n'.join(conflicts) + '\n')

    return instructions, exit_message


# ---------------------------------------------------------------------------
def test_check_instructions() -> None:
    """ Test check_instructions() """

    # Consistent instructions
    df = pd.DataFrame([['123', 'do not merge'], ['456', 'do not merge'],
                       ['789', 'do not merge, do not merge']],
                      columns=['ID', 'review_dup_urls'])

    instructions, error_message = check_instructions(df['ID'],
                                                     df['review_dup_urls'])

    assert instructions == 'do not merge'
    assert error_message == ''

    # Inconsistent instructions
    df = pd.DataFrame([['123', 'merge on record with best name prob'],
                       ['456', 'do not merge']],
                      columns=['ID', 'review_dup_urls'])

    instructions, error_message = check_instructions(df['ID'],
                                                     df['review_dup_urls'])

    assert len(instructions) == 2
    assert error_message != ''
    assert re.findall('456: "do not merge"', error_message)


# ---------------------------------------------------------------------------
def process_duplicate_urls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process rows flagged for duplicate URLs

    Parameters:
    `df`: Input dataframe

    Return: Dataframe
    """

    out_df = df[df['duplicate_urls'] == '']
    duplicate_url_df = df[df['duplicate_urls'] != '']
    exit_message = ''
    cols_for_removal = [
        'duplicate_urls', 'review_dup_urls', 'review_notes_dup_urls'
    ]

    out_df.drop(cols_for_removal, axis='columns', inplace=True)

    for _, group_df in duplicate_url_df.groupby('extracted_url'):

        instructs, problem = check_instructions(group_df['ID'],
                                                group_df['review_dup_urls'])

        if problem:
            exit_message += problem
            continue

        if instructs[0] == 'merge on record with best name prob':
            group_df = group_df.sort_values(
                'best_name_prob', ascending=False).groupby('extracted_url')
            group_df = group_df.agg({
                'ID': join_commas,
                'best_common': join_commas,
                'best_common_prob': join_commas,
                'best_full': join_commas,
                'best_full_prob': join_commas,
                'article_count': len,
                'duplicate_names': 'first',
                'review_dup_names': 'first',
                'review_notes_dup_names': 'first',
                'publication_date': 'first'
            }).reset_index()

            group_df = wrangle_names(group_df, 'best_common',
                                     'best_common_prob', 'best_full',
                                     'best_full_prob')
        else:
            group_df.drop(cols_for_removal, axis='columns', inplace=True)

        out_df = pd.concat([out_df, group_df])

    if exit_message:
        sys.exit(exit_message)

    return out_df


# ---------------------------------------------------------------------------
def test_process_duplicate_urls(raw_data: pd.DataFrame) -> None:
    """ Test process_duplicate_urls() """

    out_df = process_duplicate_urls(drop_low_probs(clean_df(raw_data)))

    # Previous rows not removed
    assert '1' in out_df['ID'].values
    assert '2' in out_df['ID'].values

    # Same URL, marked do not merge
    assert '4' in out_df['ID'].values
    assert '5' in out_df['ID'].values

    # Same URL, marked merge
    assert '7, 6' in out_df['ID'].values
    assert 'name6' not in out_df['best_name'].values
    assert 'name7' in out_df['best_name'].values

    # Manual review columns removed
    assert all(col not in out_df.columns for col in
               ['duplicate_urls', 'review_dup_urls', 'review_notes_dup_urls'])


# ---------------------------------------------------------------------------
def process_duplicate_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process rows that are flagged for having duplicate names.
    When records are merged with conflicting URLs, the newest URL is used.

    Parameters:
    `df`: Input dataframe

    Return: Dataframe
    """

    out_df = df[df['duplicate_names'] == '']
    duplicate_name_df = df[df['duplicate_names'] != '']
    exit_message = ''
    cols_for_removal = [
        'duplicate_names', 'review_dup_names', 'review_notes_dup_names'
    ]

    for _, group_df in duplicate_name_df.groupby('best_name'):

        do_not_merge = group_df[group_df['review_dup_names'] == 'do not merge']
        do_merge = group_df[group_df['review_dup_names'] != 'do not merge']

        if len(do_merge) > 0:
            _, problem = check_instructions(do_merge['ID'],
                                            do_merge['review_dup_names'])

            if problem:
                exit_message += problem
                continue

            do_merge = do_merge.sort_values(
                'publication_date', ascending=False).groupby('best_name')
            do_merge = do_merge.agg({
                'ID': join_commas,
                'best_common': join_commas,
                'best_common_prob': join_commas,
                'best_full': join_commas,
                'best_full_prob': join_commas,
                'article_count': len,
                'extracted_url': 'first',
                'publication_date': 'first'
            }).reset_index()

            do_merge = wrangle_names(do_merge, 'best_common',
                                     'best_common_prob', 'best_full',
                                     'best_full_prob')

        out_df = pd.concat([out_df, do_merge, do_not_merge])

    out_df.drop(cols_for_removal, axis='columns', inplace=True)

    if exit_message:
        sys.exit(exit_message)

    return out_df


# ---------------------------------------------------------------------------
def test_process_duplicate_names(raw_data: pd.DataFrame) -> None:
    """ Test process_duplicate_names """

    out_df = process_duplicate_names(
        process_duplicate_urls(drop_low_probs(clean_df(raw_data))))

    # Previous rows not removed
    assert '1' in out_df['ID'].values
    assert '2' in out_df['ID'].values
    assert '4' in out_df['ID'].values

    # Same name, marked do not merge
    assert '8' in out_df['ID'].values
    assert '9' in out_df['ID'].values

    # Same name, marked merge all "dup name" IDs
    assert '11, 10' in out_df['ID'].values
    assert 'url10' not in out_df['extracted_url'].values
    assert 'url11' in out_df['extracted_url'].values

    # Same name, merge only some
    assert '12' in out_df['ID'].values
    assert '14, 13' in out_df['ID'].values
    assert 'url12' in out_df['extracted_url'].values
    assert 'url14' in out_df['extracted_url'].values

    # Manual review columns removed
    assert all(
        col not in out_df.columns for col in
        ['duplicate_names', 'review_dup_names', 'review_notes_dup_names'])


# ---------------------------------------------------------------------------
def count_articles(id_list: str) -> str:
    """
    Count the number of article IDs in a string

    Parameters:
    `id_list`: String which is a list of IDs

    Return: Number of articles in list as string
    """

    return str(id_list.count(',') + 1)


# ---------------------------------------------------------------------------
def test_count_articles() -> None:
    """ Test count_articles() """

    assert count_articles('123') == '1'
    assert count_articles('123, 456') == '2'
    assert count_articles('123, 456, 789') == '3'


# ---------------------------------------------------------------------------
def update_article_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Update the count of articles after deduplication based on the number of
    IDs in the ID column

    Parameters:
    `df`: Input dataframe

    Output: Dataframe
    """

    df['article_count'] = df['ID'].map(count_articles)

    return df


# ---------------------------------------------------------------------------
def test_update_article_count() -> None:
    """ Test update_article_count """

    in_df = pd.DataFrame([['123', 'name1', '1'], ['123, 456', 'name2', '1']],
                         columns=['ID', 'best_name', 'article_count'])

    out_df = pd.DataFrame([['123', 'name1', '1'], ['123, 456', 'name2', '2']],
                          columns=['ID', 'best_name', 'article_count'])

    assert_frame_equal(update_article_count(in_df), out_df)


# ---------------------------------------------------------------------------
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process manually reviewed data.

    Parameters:
    `df`: Manually reviewed dataframe

    Return: Processed dataframe
    """

    df = clean_df(df)
    df = drop_low_probs(df)
    df = process_duplicate_urls(df)
    df = process_duplicate_names(df)
    df = update_article_count(df)

    return df


# ---------------------------------------------------------------------------
def make_filename(out_dir: str, infile_name: str) -> str:
    '''
    Make filename for output reusing input file's basename

    Parameters:
    `out_dir`: Output directory
    `infile_name`: Input file name

    Return: Output filename
    '''

    return os.path.join(out_dir, os.path.basename(infile_name))


# ---------------------------------------------------------------------------
def test_make_filenames() -> None:
    """ Test make_filenames() """

    assert make_filename(
        'out/checked_urls',
        'out/urls/predictions.csv') == ('out/checked_urls/predictions.csv')


# ---------------------------------------------------------------------------
def main() -> None:
    """ Main function """

    args = get_args()
    out_dir = args.out_dir

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    outfile = make_filename(out_dir, args.file.name)

    in_df = pd.read_csv(args.file, dtype=str).fillna('')

    check_data(in_df)

    out_df = process_data(in_df)

    out_df.to_csv(outfile, index=False)

    print(f'Done. Wrote output to {outfile}.')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
