"""
Integration tests for class_data_generator
Author: Kenneth Schackart
"""

import os
import re
import shutil
from subprocess import getstatusoutput

import test_utils as tu

PRG = 'src/class_data_generator.py'
INPUT1 = 'tests/inputs/class_data_generator/input.csv'
BAD_INPUT = 'tests/inputs/class_data_generator/bad_input.csv'
WRONG_INPUT = 'tests/inputs/ner_data_generator/input.csv'
TRAIN_OUT = 'tests/inputs/class_data_generator/train_paper_classif.csv'
VAL_OUT = 'tests/inputs/class_data_generator/val_paper_classif.csv'
TEST_OUT = 'tests/inputs/class_data_generator/test_paper_classif.csv'


# ---------------------------------------------------------------------------
def test_exists() -> None:
    """ Program exists """

    assert os.path.isfile(PRG)


# ---------------------------------------------------------------------------
def test_testing_inputs() -> None:
    """ Test that the necessary input files are present """

    for filename in [
            INPUT1, BAD_INPUT, WRONG_INPUT, TRAIN_OUT, VAL_OUT, TEST_OUT
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

    # File has duplicate rows of an ID
    retval, out = getstatusoutput(f'{PRG} {BAD_INPUT}')
    assert retval != 0
    assert re.search('not equal', out)


# ---------------------------------------------------------------------------
def test_runs_okay() -> None:
    """ Runs deterministically on valid input"""

    out_dir = tu.random_string()
    train_file = os.path.join(out_dir, 'train_paper_classif.csv')
    val_file = os.path.join(out_dir, 'val_paper_classif.csv')
    test_file = os.path.join(out_dir, 'test_paper_classif.csv')

    try:
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)

        retval, out = getstatusoutput(f'{PRG} --seed --splits 0.5 0.25 0.25'
                                      f' -o {out_dir} {INPUT1}')

        assert retval == 0
        assert re.search(f'Done. Wrote 3 files to {out_dir}.', out)
        assert os.path.isdir(out_dir)
        for filename in [train_file, val_file, test_file]:
            assert os.path.isfile(filename)

        for out_file, expected_out in zip([train_file, val_file, test_file],
                                          [TRAIN_OUT, VAL_OUT, TEST_OUT]):
            assert open(out_file).read() == open(expected_out).read()

    finally:
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
