""" Tests """

import os
import re
from subprocess import getstatusoutput

import test_utils as tu

# import shutil

PRG = 'src/train.py'


# ---------------------------------------------------------------------------
def test_exists() -> None:
    """ Program exists """

    assert os.path.isfile(PRG)


# ---------------------------------------------------------------------------
def test_usage() -> None:
    """ Usage """

    for flag in ['-h', '--help']:
        retval, out = getstatusoutput(f'{PRG} {flag}')
        assert retval == 0
        assert out.lower().startswith('usage')


# ---------------------------------------------------------------------------
def run_bad_file(flag: str) -> None:
    """ Dies on bad input file """

    bad = tu.random_string()

    retval, out = getstatusoutput(f'{PRG} {flag} {bad}')
    assert retval != 0
    assert out.lower().startswith('usage:')
    assert re.search(f"No such file or directory: '{bad}'", out)


# ---------------------------------------------------------------------------
def test_bad_input_files() -> None:
    """ Dies on bad input files """

    flags = ['-t', '--train-file', '-v', '--val-file']

    for flag in flags:
        run_bad_file(flag)


# ---------------------------------------------------------------------------
def test_bad_model() -> None:
    """ Dies on bad model choice """

    tu.bad_model(PRG)


# ---------------------------------------------------------------------------
def test_bad_predictor() -> None:
    """ Dies on bad predictor choice """

    tu.bad_predictor(PRG)


# ---------------------------------------------------------------------------
def test_bad_descriptors() -> None:
    """ Incorrect number of descriptive labels """

    tu.bad_descriptors(PRG)
