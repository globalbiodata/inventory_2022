# Overview

This directory contains the source code used in this project. The process for both article classification and NER are very similar, so those steps for each process are described together.

These scripts are modules, and are not to be executed:
* `class_data_handler.py`
* `ner_data_handler.py`
* `utils.py`

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
* `query_epmc.py`
* `utils.py`

## Accessing Help

Each of the executable scripts listed above will respond to the `-h` or `--help` flag by providing a usage statement.

For example:
```
$ url_extractor.py --help
usage: url_extractor.py [-h] [-o DIR] FILE

Extract URLs from "text" column of file.

positional arguments:
  FILE               Input file (csv)

optional arguments:
  -h, --help         show this help message and exit
  -o, --out-dir DIR  Output directory (default: out/)
```

# Running Query

`query_epmc.py`

EuropePMC is queried using the query provided. The query can be supplied directly in the command-line (place quotes around it), or can be the name of a file whose only content is the query string. Such a file exists in [config/query.txt](../config/query.txt).

The query should have the placeholders {0} and {1} for the publication date ranges. This makes the query reuable, and the `-f|--from-date` and `-t|--to-date` are provided at runtime. Again, these can be provided as literal strings, or as text files.

Dates can be formatted as any of the following: 

* YYYY
* YYYY-MM
* YYYY-MM-DD

If the query has no placeholders, the `--from-date` and `--to-date` arguments are ignored.


Once the query is completed two files are created in `--out-dir`:

* `last_query_date.txt`: File with the `--to-date`, defaulting to today's date
* `new_query_results.csv`: Containing IDs, titles, and abstracts from query

# Data Generation

`class_data_generator.py` and `ner_data_generator.py`

The first step for training is processing the manually curated files of labeled data. This includes splitting into training, validation, and testing splits. The proportions assigned to train, val, test splits can be specified with the `--splits` arguement. To make the splits reproducible, the `-r|--seed` flag can be used to make the split non-random and consistent.

Both scripts output 3 .csv files containing the split data.

`ner_data_generator.py` outputs 3 additional files (.pkl), which are the inputs to `ner_train.py`. These files contain the tagged tokens for training.


# Model training

`class_train.py` and `ner_train.py`

These scripts load a pretrained `--model` from HuggingFace, and perform fine-tuning and classifier training. Training is done using the train and val splits from [Data Generation](#Data-Generation). `class_train.py` takes .csv files, while `ner_train.py` takes .pkl files.

The `-m|--model-name` must be a valid HuggingFace model name, such as those in the "hf_name" column  of [the model configuration file](../config/models_info.tsv).

Several training parameters can be changed, such as learning rate, weight decay, batch size, and number of epochs. A learning rate scheduler can be optionally used.

If it is desired to run training on only a certain number of samples, the `-nt|--num-training` argument can be used.

Finally, to make training reproducible, the `-r|--seed` option is available.

During each epoch of model training, *F*1 score is computed for predictions on the validation set. Once validation *F*1 begins to drop, training is ended, since this indicates that over-fitting has begun. If the validation *F*1 score does not drop, training will continue until `-ne|--num-epochs` is met.

Once training is complete, two outputs are created in `--out-dir`:
* `checkpoint.pt`: The trained model checkpoint, which can be used for prediction
* `train_stats.csv`: File containing model performance statistics for each epoch of training.

# Model selection

`model_picker.py`

Once all models that are to be compared have finished training, `model_picker.py` takes all the `train_stats.csv` files as input in order to select the one with the highest validation *F*1 score.

Two outputs are created in `--out-dir`:
* `{best_model_name}/best_checkpt.pt`: Checkpoint of best model (copied from its original location)
* `{best_model_name}/combined_stats.csv`: File with training stats for each epoch of every model compared

# Model evaluation

`class_final_eval.py` and `ner_final_eval.py`

Final evaluation of the chosen models is performed using `class_final_eval.py` and `ner_final_eval.py` on the witheld test sets. Precision, recall, *F*1 and loss are computed.

One output file is created in `--out-dir`:
* `{outdir}/metrics.csv`

# Prediction

`class_predict.py` and `ner_predict.py`

The trained model checkpoint is used to perform prediction. NER prediction should only be performed on articles predicted to be (or manually classified as) biodata resources.

# Downstream tasks

Once classification and NER have been performed, other information can be gathered about the predicted resources. These next steps take as input the output from `ner_predict.py`.

## URL extraction

`url_extractor.py` is used to extract all unique URLs from the "text" (title + abstract). This is done using a regular expression.