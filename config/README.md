# Project Configuration Files

This directory contains configuration files for several aspects of this project.

```
.
├── .pylintrc       # Configurations for pylint
├── config.yml      # Snakemake pipeline configs
├── environment.yml # Conda environment description
├── models_info.tsv # Model training parameters
├── query.txt       # EuropePMC query string
└── README.md
```

# File Descriptions

## `.pylintrc`

Since the test suite includes linting of all Python files with pylint, this configuration file informs pylint about what rules to follow during linting. This helps ensure a consistent testing environment across machines.

## `config.yml`

This YAML file contains the majority of the configurations used in the Snakemake pipelines, such as directories and some model training configurations. Changes can be made to this file to alter the parameters used by the Snakemake pipelines.

## `environment.yml`

This YAML file can be used to directly create a conda environment with all of the dependencies of this project.

## `models_info.tsv`

This tab-separated file contains the configurations used during model training. The columns of this file are as follows:

| model | hf_name | batch_size | learning_rate | weight_decay | scheduler
| :-: | :-: | :-: | :-: | :-: | :-: |
unique, shortened model name used for convenience | pretrained model name as it appears in HuggingFace Hub | number of training examples used by one processor in one training step | step size at each iteration while moving toward a minimum of a loss function | weight decay (L2 penalty) | optional learning rate scheduler flag ( `-lr` or empty)

New rows can be added to this file, and the training pipeline re-run to evaluate the new models against the others. Multiple rows for a given file can also be added to compare performance of training with certain parameters, but the model column should remain unique.

More information about these parameters can be found on [Hugging Face 🤗](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)

## `query.txt`

This text file contains a single string, which is the search query sent to EuropePMC. The publication date range should contain placeholders `{0}` and `{1}`, for the from- and to-dates respectively. If placeholders are not used, the date arguments of `src/query_epmc.py` are ignored.

