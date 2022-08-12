# Modules

This directory contains Python modules. The contents of the files present here can be imported into other Python scripts or modules

```sh
.
├── __init__.py           # Allows for modules to be imported
├── aliases.py            # Type aliases
├── class_data_handler.py # Preparing classification data
├── constants.py          # Constants
├── custom_classes.py     # Custom classes
├── filings.py            # Reading/writing files
├── metrics.py            # Performance metrics
├── ner_data_handler.py   # Preparing NER data
├── runtime.py            # Changing or assessing runtime
└── wrangling.py          # Wrangling/cleaning data structures
```

## `__init__.py`

Empty file that is necessary for importing these modules from other places in this repository.

## `aliases.py`

Custom type aliases that simplify type annotations.

## `class_data_handler.py`

Modules mostly used by [../class_data_generator.py](../class_data_generator.py), for labeling, creating dataloaders and tokenizing.

## `constants`

Constant values, such as dictionaries mapping text to integer labels for NER annotation.

## `custom_classes`

Custom data structures (classes). These are generally just classes which inherit from `typing.NamedTuple`. They do not have custom methods and are just for simplifying how data are passed and structured.

## `filing.py`

Functions for reading files, creating file names, and saving files.

## `metrics.py`

Functions for calculcating and working with performance metrics.

## `ner_data_handler.py`

Modules mostly used by [../ner_data_generator.py](../ner_data_generator.py), for labeling, creating dataloaders and tokenizing.

## `runtime.py`

Functions for modifying the runtime environment (such as setting global random seeds) or detecting the environment (such as determining if CPU or CUDA should be used by `torch`).

## `wrangling.py`

Functions for wrangling and cleaning data, especially text.
