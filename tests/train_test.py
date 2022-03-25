""" Tests """

import os
import random
import re
# import shutil
import string
from subprocess import getstatusoutput

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

    bad = random_string()

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
def run_bad_option(flag: str) -> None:
    """ Dies on bad option """

    bad = random_string()

    retval, out = getstatusoutput(f'{PRG} {flag} {bad}')
    assert retval != 0
    assert out.lower().startswith('usage:')
    assert re.search('invalid choice', out)
    assert re.search(bad, out)
    assert re.search('choose from', out)


# ---------------------------------------------------------------------------
def test_bad_model() -> None:
    """ Dies on bad model choice """

    run_bad_option('--model-name')


# ---------------------------------------------------------------------------
def test_bad_predictor() -> None:
    """ Dies on bad predictor choice """

    run_bad_option('--predictive-field')


# ---------------------------------------------------------------------------
def test_bad_descriptors() -> None:
    """ Incorrect number of descriptive labels """

    label = random_string()

    retval, out = getstatusoutput(f'{PRG} -desc {label}')
    assert retval != 0
    assert out.lower().startswith('usage:')
    assert re.search('-desc/--descriptive-labels', out)
    assert re.search('expected 2 arguments', out)


# ---------------------------------------------------------------------------
def random_string() -> str:
    """ Generate a random string """

    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
