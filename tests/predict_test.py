""" Tests """

import os
import re
# import shutil
from subprocess import getstatusoutput

import test_utils as tu

PRG = 'src/class/predict.py'
INPUT1 = 'tests/inputs/example_data.csv'


# ---------------------------------------------------------------------------
def test_exists():
    """ Program exists """

    assert os.path.isfile(PRG)


# ---------------------------------------------------------------------------
def test_usage():
    """ Usage """

    for flag in ['-h', '--help']:
        retval, out = getstatusoutput(f'{PRG} {flag}')
        assert retval == 0
        assert out.lower().startswith('usage')


# ---------------------------------------------------------------------------
def test_bad_file() -> None:
    """ Dies on bad input file """

    bad = tu.random_string()

    for flag in ['-i', '--input-file']:
        retval, out = getstatusoutput(f'{PRG} {flag} {bad}')
        assert retval != 0
        assert out.lower().startswith('usage:')
        assert re.search(f"No such file or directory: '{bad}'", out)


# ---------------------------------------------------------------------------
def test_bad_checkpoint() -> None:
    """ Dies on bad model checkpoint """

    bad = tu.random_string()

    for flag in ['-c', '--checkpoint']:
        retval, out = getstatusoutput(f'{PRG} {flag} {bad} -i {INPUT1}')
        assert retval != 0
        assert out.lower().startswith('usage:')
        assert re.search(f"No such file or directory: '{bad}'", out)


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
