"""
Integration tests for ner_data_generator
Author: Kenneth Schackart
"""

import os
import re
import shutil
from subprocess import getstatusoutput

import test_utils as tu

PRG = 'src/ner_data_generator.py'
INPUT1 = 'tests/inputs/ner_data_generator/input.csv'
WRONG_INPUT = 'tests/inputs/class_data_generator/input.csv'
TRAIN_OUT_CSV = 'tests/inputs/ner_data_generator/train_ner.csv'
VAL_OUT_CSV = 'tests/inputs/ner_data_generator/val_ner.csv'
TEST_OUT_CSV = 'tests/inputs/ner_data_generator/test_ner.csv'
TRAIN_OUT_PKL = 'tests/inputs/ner_data_generator/train_ner.pkl'
VAL_OUT_PKL = 'tests/inputs/ner_data_generator/val_ner.pkl'
TEST_OUT_PKL = 'tests/inputs/ner_data_generator/test_ner.pkl'


# ---------------------------------------------------------------------------
def test_exists() -> None:
    """ Program exists """

    assert os.path.isfile(PRG)


# ---------------------------------------------------------------------------
def test_testing_inputs() -> None:
    """ Test that the necessary input files are present """

    for filename in [
            INPUT1, WRONG_INPUT, TRAIN_OUT_CSV, VAL_OUT_CSV, TEST_OUT_CSV,
            TRAIN_OUT_PKL, VAL_OUT_PKL, TEST_OUT_PKL
    ]:
        assert os.path.isfile(filename)


# ---------------------------------------------------------------------------
def test_usage() -> None:
    """ Usage """

    for flag in ['-h', '--help']:
        retval, out = getstatusoutput(f'{PRG} {flag}')
        assert retval == 0
        assert out.lower().startswith('usage')


# ---------------------------------------------------------------------------
def test_missing_input_file() -> None:
    """ Dies when input file does not exist """

    bad = tu.random_string()

    retval, out = getstatusoutput(f'{PRG} {bad}')
    assert retval != 0
    assert out.lower().startswith('usage:')
    assert re.search(f"No such file or directory: '{bad}'", out)


# ---------------------------------------------------------------------------
def test_bad_splits() -> None:
    """ Dies with bad splits argument """

    # Incorrect number of splits
    retval, out = getstatusoutput(f'{PRG} {INPUT1} --splits 0.5 0.5')
    assert retval != 0
    assert out.lower().startswith('usage:')
    assert re.search('--splits: expected 3 arguments', out)

    # Splits don't add to 1
    retval, out = getstatusoutput(f'{PRG} {INPUT1} --splits 0.5 0.25 0.1')
    assert retval != 0
    assert out.lower().startswith('usage:')
    assert re.search('must sum to 1', out)


# ---------------------------------------------------------------------------
def test_bad_input() -> None:
    """ Dies on bad input file """

    # Wrong file, necessary columns not present
    retval, out = getstatusoutput(f'{PRG} {WRONG_INPUT}')
    assert retval != 0
    assert re.search('Input data does not have the expected columns', out)


# ---------------------------------------------------------------------------
def test_runs_okay() -> None:
    """ Runs deterministically on valid input"""

    out_dir = tu.random_string()
    train_csv = os.path.join(out_dir, 'train_ner.csv')
    val_csv = os.path.join(out_dir, 'val_ner.csv')
    test_csv = os.path.join(out_dir, 'test_ner.csv')
    train_pkl = os.path.join(out_dir, 'train_ner.pkl')
    val_pkl = os.path.join(out_dir, 'val_ner.pkl')
    test_pkl = os.path.join(out_dir, 'test_ner.pkl')

    try:
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)

        retval, out = getstatusoutput(f'{PRG} --seed --splits 0.8 0.1 0.1'
                                      f' -o {out_dir} {INPUT1}')

        assert retval == 0
        assert re.search(f'Done. Wrote 6 files to {out_dir}.', out)
        assert os.path.isdir(out_dir)
        for filename in [
                train_csv, val_csv, test_csv, train_pkl, val_pkl, test_pkl
        ]:
            assert os.path.isfile(filename)

        expected_outs = [TRAIN_OUT_CSV, VAL_OUT_CSV, TEST_OUT_CSV]
        actual_outs = [train_csv, val_csv, test_csv]
        for out_file, expected_out in zip(expected_outs, actual_outs):
            assert open(out_file).read() == open(expected_out).read()

    finally:
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
