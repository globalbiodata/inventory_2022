"""
Purpose: Module of constants
Authors: Ana-Maria Istrate and Kenneth Schackart
"""

import sys

# ---------------------------------------------------------------------------
NER_TAG2ID = {'O': 0, 'B-COM': 1, 'I-COM': 2, 'B-FUL': 3, 'I-FUL': 4}
"""
Mapping of NER tags to numerical labels

`key`: String NER label ("O", "B-COM", "I-COM", "B-FUL", "I-FUL")
`value`: Numerical label (1, 2, 3, 4)
"""

ID2NER_TAG = {v: k for k, v in NER_TAG2ID.items()}
"""
Mapping of numerical labels to NER tags

`key`: Numerical label (1, 2, 3, 4)
`value`: String NER label ("O", "B-COM", "I-COM", "B-FUL", "I-FUL")
"""

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    sys.exit('This file is a module, and is not meant to be run.')
