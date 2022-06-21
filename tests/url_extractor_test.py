""" Tests """

import os
import re
import shutil
from subprocess import getstatusoutput

import test_utils as tu

PRG = 'src/url_extractor.py'
INPUT = 'tests/inputs/url_extractor/ner_output.csv'
OUTPUT = 'tests/inputs/url_extractor/url_output.csv'


# ---------------------------------------------------------------------------
def test_exists() -> None:
    """ Program exists """

    assert os.path.isfile(PRG)


# ---------------------------------------------------------------------------
def test_testing_inputs() -> None:
    """ Test that the necessary input files are present """

    for filename in [INPUT, OUTPUT]:
        assert os.path.isfile(filename)


# ---------------------------------------------------------------------------
def test_usage() -> None:
    """ Usage """

    for flag in ['-h', '--help']:
        retval, out = getstatusoutput(f'{PRG} {flag}')
        assert retval == 0
        assert out.lower().startswith('usage')


# ---------------------------------------------------------------------------
def test_bad_input_file() -> None:
    """ Dies on bad input file """

    bad = tu.random_string()

    retval, out = getstatusoutput(f'{PRG} {bad}')
    assert retval != 0
    assert out.lower().startswith('usage:')
    assert re.search(f"No such file or directory: '{bad}'", out)


# ---------------------------------------------------------------------------
def test_runs_okay() -> None:
    """ Runs on good input """

    out_dir = tu.random_string()

    try:
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)

        rv, out = getstatusoutput(f'{PRG} -o {out_dir} {INPUT}')

        assert rv == 0
        out_file = os.path.join(out_dir, 'ner_output.csv')
        assert out == (f'Done. Wrote output to {out_file}.')
        assert os.path.isdir(out_dir)
        assert os.path.isfile(out_file)
        assert open(out_file).read() == open(OUTPUT).read()

    finally:
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
