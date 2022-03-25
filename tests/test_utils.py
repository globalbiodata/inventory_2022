"""
Utilities for testing
"""

import random
import re
import string
from subprocess import getstatusoutput


# ---------------------------------------------------------------------------
def random_string() -> str:
    """ Generate a random string """

    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))


# ---------------------------------------------------------------------------
def run_bad_option(prg: str, flag: str) -> None:
    """ Dies on bad option """

    bad = random_string()

    retval, out = getstatusoutput(f'{prg} {flag} {bad}')
    assert retval != 0
    assert out.lower().startswith('usage:')
    assert re.search('invalid choice', out)
    assert re.search(bad, out)
    assert re.search('choose from', out)


# ---------------------------------------------------------------------------
def bad_model(prg: str) -> None:
    """ Dies on bad model choice """

    run_bad_option(prg, '--model-name')


# ---------------------------------------------------------------------------
def bad_predictor(prg: str) -> None:
    """ Dies on bad predictor choice """

    run_bad_option(prg, '--predictive-field')


# ---------------------------------------------------------------------------
def bad_descriptors(prg: str) -> None:
    """ Incorrect number of descriptive labels """

    label = random_string()

    retval, out = getstatusoutput(f'{prg} -desc {label}')
    assert retval != 0
    assert out.lower().startswith('usage:')
    assert re.search('-desc/--descriptive-labels', out)
    assert re.search('expected 2 arguments', out)
