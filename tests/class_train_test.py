""" Tests """

import os
import re
from subprocess import getstatusoutput

import test_utils as tu

# import shutil

PRG = 'src/class_train.py'


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


# ---------------------------------------------------------------------------
def test_bad_numeric_args() -> None:
    """ Incorrect type for numeric arguments """

    bad_float = tu.random_string()

    for flag in ['-r', '-decay']:
        retval, out = getstatusoutput(f'{PRG} {flag} {bad_float}')
        assert retval != 0
        assert out.lower().startswith('usage')
        assert re.search(f"invalid float value: '{bad_float}'", out)

    bad_int = tu.random_float()

    for flag in ['-max', '-nt', '-ne', '-batch']:
        retval, out = getstatusoutput(f'{PRG} {flag} {bad_int}')
        assert retval != 0
        assert out.lower().startswith('usage')
        assert re.search(f"invalid int value: '{bad_int}'", out)
