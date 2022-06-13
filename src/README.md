# Overview

This directory contains the source code used in this project. The process for both article classification and NER are very similar, so those steps for each process are described together.

The following are executable Python scripts:
* `class_data_generator.py`
* `class_predict.py`
* `class_train.py`
* `model_picker.py`
* `ner_data_generator.py`
* `ner_predict.py`
* `ner_train.py`
* `url_extractor.py`
* `class_predict.py`
* `utils.py`

While these scripts are not to be executed, and are to supply shared functions:
* `class_data_handler.py`
* `ner_data_handler.py`
* `utils.py`

## Accessing Help

Each of the executable scripts listed above will respond to the `-h` or `--help` flag by providing a usage statement.

For example:
```
$ url_extractor.py --help
usage: url_extractor.py [-h] [-o DIR] FILE

Extract URLs from text. File must have a column named "text".

positional arguments:
  FILE               Input file (csv)

optional arguments:
  -h, --help         show this help message and exit
  -o, --out-dir DIR  Output directory (default: out/)
```

# Data Generation

`class_data_generator.py` and `ner_data_generator.py`

The first step for training is processing the manually curated files of labeled data. This includes splitting into training, validation, and testing splits. The proportions assigned to train, val, test splits can be specified with the `--splits` arguement. To make the splits reproducible, the `-r|--seed` flag can be used to make the split non-random and consistent.

Both scripts output 3 .csv files containing the split data.

`ner_data_generator.py` outputs 3 additional files (.pkl), which are the inputs to `ner_train.py`. These files contain the tagged tokens for training.


# Model training

`class_train.py` and `ner_train.py`

These scripts load a pretrained `--model` from HuggingFace, and perform fine-tuning and classifier training. Training is done using the train and val splits from [Data Generation](#Data-Generation). `class_train.py` takes .csv files, while `ner_train.py` takes .pkl files.

The `-m|--model_name` must be a valid HuggingFace model name, such as those in the "hf_name" column  of [the model configuration file](../config/models_info.tsv).

Several training parameters can be changed, such as learning rate, weight decay, batch size, and number of epochs. A learning rate scheduler can be optionally used.

If it is desired to run training on only a certain number of samples, the `-nt|--num_training` argument can be used.

Finally, to make training reproducible, the `-r|--seed` option is available.

During each epoch of model training, *F*1 score is computed for predictions on the validation set. Once validation *F*1 begins to drop, training is ended, since this indicates that over-fitting has begun. If the validation *F*1 score does not drop, training will continue until `-ne|--num_epochs` is met.

Once training is complete, two outputs are created in `--outdir`:
* `checkpoint.pt`: The trained model checkpoint, which can be used for prediction
* `train_stats.csv`: File containing model performance statistics for each epoch of training.

# Model selection

`model_picker.py`

Once all models that are to be compared have finished training, `model_picker.py` takes all the `train_stats.csv` files as input in order to select the one with the highest validation *F*1 score.

Two outputs are created in `--outdir`:
* `{best_model_name}/best_checkpt.pt`: Checkpoint of best model (copied from its original location)
* `{best_model_name}/combined_stats.csv`: File with training stats for each epoch of every model compared

# Prediction

`class_predict.py` and `ner_predict.py`

The trained model checkpoint is used to perform prediction. NER prediction should only be performed on articles predicted to be (or manually classified as) biodata resources.

# Downstream tasks

Once classification and NER have been performed, other information can be gathered about the predicted resources. These next steps take as input the output from `ner_predict.py`.

## URL extraction

`url_extractor.py` is used to extract all unique URLs from the "text" (title + abstract). This is done using a regular expression.