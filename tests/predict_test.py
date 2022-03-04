""" Tests """

import os
# import random
# import re
# import shutil
# import string
from subprocess import getstatusoutput

PRG = './predict.py'


# ---------------------------------------------------------------------------
def test_exists():
    """ Program exists """

    assert os.path.isfile(PRG)


# ---------------------------------------------------------------------------
def test_usage():
    """ Usage """

    for flag in ['-h', '--help']:
        retval, out = getstatusoutput(f'python3 {PRG} {flag}')
        assert retval == 0
        assert out.lower().startswith('usage')
