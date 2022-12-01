# Overview

This directory contains the source code used in this project.

```sh
.
├── inventory_utils/         # Modules used in the project
├── check_urls.py            # Gather information from URLs
├── class_data_generator.py  # Prepare and split classification data
├── class_final_eval.ppy     # Evaluate best model on test set
├── class_predict.py         # Use trained model to predict classification
├── class_train.py           # Train classifier
├── combine_stats.py         # Combine training/evulation stats files
├── flag_for_review.py       # Flag inventory for manual review
├── initial_deduplication.py # Perform initial automated deduplication
├── model_picker.py          # Select best trained model
├── ner_data_generator.py    # Prepare and split NER data
├── ner_final_eval.py        # Evaluate best model on test set
├── ner_predict.py           # Use trained model to perform NER
├── ner_train.py             # Train NER model
├── process_countries.py     # Process informationa about country codes
├── process_manual_review.py # Process manually reviewed inventory
├── process_names.py         # Process predicted names to determine best
├── query_epmc.py            # Query EuropePMC
└── url_extractor.py         # Extract URLs from text
```

## Accessing Help

Each of the executable scripts listed above will respond to the `-h` or `--help` flag by providing a usage statement.

For example:
```sh
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

Several training parameters can be changed, such as learning rate, weight decay, batch size, and number of epochs. A learning rate scheduler can be optionally used. See [../config/README.csv](../config/README.md#modelsinfotsv) for more information on these parameters.

If it is desired to run training on only a certain number of samples, the `-nt|--num-training` argument can be used.

Finally, to make training reproducible, the `-r|--seed` option is available.

During each epoch of model training, *F*1 score is computed for predictions on the validation set. Once validation *F*1 begins to drop, training is ended, since this indicates that over-fitting has begun. If the validation *F*1 score does not drop, training will continue until `-ne|--num-epochs` is met.

Once training is complete, two outputs are created in `--out-dir`:
* `checkpoint.pt`: The trained model checkpoint, which can be used for prediction
* `train_stats.csv`: File containing model performance statistics for each epoch of training.

# Model selection

`model_picker.py`

Once all models that are to be compared have finished training, `model_picker.py` takes all the model checkpoint files as input in order to select the one with the highest validation score. Which metric to use for choosing the best model is passed in as `-m|--metric`.

One output is created in `--out-dir`:
* best_checkpt.txt`: Text file containing locations of best model checkpoint.

# Model evaluation

`class_final_eval.py` and `ner_final_eval.py`

Final evaluation of the chosen models is performed using `class_final_eval.py` and `ner_final_eval.py` on the witheld test sets. Precision, recall, *F*1 and loss are computed.

One output file is created in `--out-dir`:
* `{outdir}/metrics.csv`

# Prediction

`class_predict.py` and `ner_predict.py`

The trained model checkpoint is used to perform prediction. NER prediction should only be performed on articles predicted to be (or manually classified as) biodata resources.

Any records for which no names were predicted are dropped by the `ner_predict.py` script.

# Downstream tasks

Once classification and NER have been performed, other information can be gathered about the predicted resources. These next steps take as input the output from `ner_predict.py`.

## URL extraction

`url_extractor.py` is used to extract all unique URLs from the "text" (title + abstract). This is done using a regular expression.

Any records with either no detected URLs or more URLs than the limit `-x|--max-urls` (default: 2) are dropped from the inventory at this stage.

## Name Selection

The NER model may predict multiple common and full names. `process_names.py` selects the common name and full name with the highest probability, as well as the name with the overall highest probability.

## Initial Deduplication

Since many resources publish articles periodically to provide updates, many records may describe the same resource. To deduplicate the inventory, an initial automated deduplication is performed.

`initial_deduplication.py` merges records that have the same `best_name` (name with highest probability) and same extracted URL (ignoring differences due to trailing slashes or difference between "http:" vs "https:").

Articles whose best_name probability is below the threshold (default: 0.978) are not merged, even if they appear to be duplicates. For this reason, it is best to use the same probability threshold for this step and the next step (flagging for manual review).

## Flagging for Manual Review

During this step, the inventory is marked for manual review.

Records with a `best_name` probability below the threshold `-p|--min-prob` (default: 0.978) are marked for review in the `low_best_prob` column. 

Articles that have the same `best_name` are maked in the `duplicate_names` column. The values in the column are the IDs of the records that have the same name as the given record.

Articles that have the same URL are marked in the `duplicate_urls` column. The values are given similar to the `duplicate_names` column.

## Processing Manually Reviewed Inventory

Once the flagged inventory has been manually reviewed, the determinations made during review are executed (*e.g.* removing certain rows, merging duplicates) by `process_manual_review.py`.

There are quite a few validations to ensure that the manual review process was conducted in a way that it can be properly processed. If there any errors are discovered during this evaluation, an error message with the ID values of bad rows will be given, as well as a description of the problem(s).

The `text` column is dropped during this step.

## Checking URLs

`check_urls.py` checks each extracted URL by submitting a request. The status of the request (either a status code or the returned error message if an exception occurs) is recorded in a new column labeled extracted_url_status. Rows without URLs are removed, since the inventory requires a URL to identify the resource.

The number of attempts to request the URL can be modified with the `-n|--num-tries` flag. To avoid exceeding the allowable number of attempts in a certain period of time, the `-b|--backoff` flag is used, where 0 adds no wait time and 1 adds the most wait time.

Additionally, for URLs that return a status less than 400, various APIs are queried attempting to obtain the geolocation of the IP address which responded to the request. From this, the country and lat, lon coordinates are recorded.

Then, each URL is submitted to [Internet Archive WaybackMachine](https://archive.org/help/wayback_api.php) to see if there exists an archived snapshot of the given URL. If so, this is marked as the checked URL.

Since this process can take quite a while, it is implemented to allow for asynchronous parallelization. Each core supplied can submit a request at the same time, and as soon as one core finishes, it submits another. By default all available cores are used, but the desired number of cores can be specified with the `-c|--cores` flag.

Additionally, a `-v|--verbose` flag is available for debugging.

*Note* Actual wait time is calculated as

```
{backoff factor} * (2 ** ({number of total retries} - 1))
```

So with a back off factor of 0.1, it will sleep for [0.0s, 0.2s, 0.4s, ...]. More information can be found in the [urllib3 documentation](https://urllib3.readthedocs.io/en/latest/reference/urllib3.util.html#module-urllib3.util.retry)

## Getting Metadata from EuropePMC

`get_meta.py` queries EuropePMC to gather further metadata on the IDs that are now in the final inventory.

In particular, the following columns are created:
`affiliation`: Author affiliations
`authors`: Author names
`grant_ids`: Grant IDs
`grant_agencies`: Unmodified grant agency list
`num_citations`: Number of citations for the paper(s) describing that resource

Since many resources are described in multiple articles, this information has to be aggregated. To do so, the IDs are separated and queried indenpendently. The information for each resource is then re-joined. All information except for number of citations is just concatenated, using a comma and space to separate the information from each article. The number of citations is summed across the number of citations for each article associated with a given resource.

The `-s|--chunk-size` parameter determines how many IDs to send to EuropePMC per request. 20  (default) is generally a good number.

## Process Country Information

`process_countries.py` porocesses and harmonizes the various information about countries in the inventory.

During URL checking, the country of the IP address is queried using various APIs, which can give differently formatted country information (US vs USA). There are mentions of countries in the author affiliations.

The Python library `pycountry` is used to extract the country information, and harmonize any differences.

The output country code can be of 4 types as defined by [ISO 3166](https://en.wikipedia.org/wiki/ISO_3166). The `-f|--format` option can be used to select the output country code type from `alpha-2`, `alpha-3`, `full`, and `numeric`.

The `extracted_url_country` column gets overwritten, by replacing country codes with the desired format.

A new column is added, `affiliation_countries`. This is created by searching for any mentions of countries in the `affiliation` column, which could by in `alpha-2`, `alpha-3`, or `full` format. All country mentions that are found are formatted in the desired countryt code format, and joined with a comma and a space.

*Note*: Both of these country information columns may not be 100% accurate. The `extracted_url_country` may not be the actual host country. The `affiliation_countries` will miss any countries that are not in one of the 3 standard formats. Additionally, there may be false positives, such as the state of Georgia being labeled as the country Georgia. If the values in the two columns match, that may be a decent indicator that they are accurate.

# Manual Workflow Examples

Here, you can find an example of how to run the entire workflow(s) manually from the command-line. This should not be necessary, since there are Snakemake pipelines to automate, and notebooks to guide, the process (see [../README.md](../README.md)). What is shown here is essentially what is run by the Snakemake pipelines. This may be useful for debugging.

*All commands shown here should be run from the root of the repository* (not from the `src/` folder).

## Training and Prediction

### Data Splitting

First, split the manually curated datasets. We will split into 80% training, 10% validation, 10% test. A random seed is used to ensure that the splits are the same each time this step is run. The choice of output directoriy is arbitrary. In these examples I will follow the schemes used in the Snakemake pipelines.
```sh
$ python3 src/class_data_generator.py \
    --out-dir out/classif_splits \
    --splits 0.8 0.1 0.1 \
    --seed \
    data/manual_classifications.csv
Done. Wrote 3 files to out/classif_splits.

$ python3 src/ner_data_generator.py \
    --out-dir out/ner_splits \
    --splits 0.8 0.1 0.1 \
    --seed \
    data/manual_ner_extraction.csv
Done. Wrote 6 files to out/ner_splits.
```

3 files are created by `class_data_generator.py`, each is a .csv file of the corresponding dataset split. 
```sh
$ ls out/classif_splits
test_paper_classif.csv  train_paper_classif.csv  val_paper_classif.csv
```

`ner_data_generator.py` creates 6 files. For each split, 2 files are created: a .csv file and .pkl file. The .pkl file is created because that is the input to the NER training. pkl is a Python Pickle file, which is essentially a way of directly storing a Python object. By storing the object directly, it simplifies reading in the tokenized and annotated data for training.
```sh
$ ls out/ner_splits
test_ner.csv  test_ner.pkl  train_ner.csv  train_ner.pkl  val_ner.csv  val_ner.pkl
```

### Model Training

Now, training can be performed. For the original project, 15 models were trained for each task (see [../config/models_info.tsv](../config/models_info.tsv) for all the models and their training parameters). For the sake of brevity, I will only demonstrate training two models for each task.

During training, several messages will be output to the terminal. They are ommitted here.

First, training the paper classifier:
```sh
$ python3 src/class_train.py \
    --train-file out/classif_splits/train_paper_classif.csv \
    --val-file out/classif_splits/val_paper_classif.csv \
    --model-name bert-base-uncased \
    --out-dir out/classif_train_out/bert \
    --num-epochs 10 \
    --batch-size 16 \
    --learning-rate 3e-5 \
    --weight-decay 0 \
    --seed

$ python3 src/class_train.py \
    --train-file out/classif_splits/train_paper_classif.csv \
    --val-file out/classif_splits/val_paper_classif.csv \
    --model-name allenai/biomed_roberta_base \
    --out-dir out/classif_train_out/biomed_roberta \
    --num-epochs 10 \
    --batch-size 16 \
    --learning-rate 2e-5 \
    --weight-decay 0 \
    --seed
```

After training, two files are created. The model checkpoint, which contains the trained model (along with training metrics), and a .csv file of the performance metrics on the training and validation sets for each epoch of training. The best performing model checkpoint is saved, even if at later epochs the performance drops.
```sh
$ ls out/classif_train_out/bert
checkpt.pt  train_stats.csv
```

Then triaining the NER model:
```sh
$ python3 src/ner_train.py \
    --train-file out/ner_splits/train_ner.pkl \
    --val-file out/ner_splits/val_ner.pkl \
    --model-name bert-base-uncased \
    --out-dir out/ner_train_out/bert \
    --num-epochs 10 \
    --batch-size 16 \
    --learning-rate 3e-5 \
    --weight-decay 0 \
    --seed

$ python3 src/ner_train.py \
    --train-file out/ner_splits/train_ner.pkl \
    --val-file out/ner_splits/val_ner.pkl \
    --model-name allenai/biomed_roberta_base \
    --out-dir out/ner_train_out/biomed_roberta \
    --num-epochs 10 \
    --batch-size 16 \
    --learning-rate 2e-5 \
    --weight-decay 0 \
    --seed
```

### Model Comparison

The same program is used to choose both the best classification and NER model. It takes any number of model checkpoints as input.

```sh
$ python3 src/model_picker.py \
    --out-dir out/classif_train_out/best \
    out/classif_train_out/*/checkpt.pt
Checkpoint of best model is out/classif_train_out/biomed_roberta/checkpt.pt
Done. Wrote output to out/classif_train_out/best/best_checkpt.txt

$ python3 src/model_picker.py \
    --out-dir out/ner_train_out/best \
    out/ner_train_out/*/checkpt.pt
Checkpoint of best model is out/ner_train_out/biomed_roberta/checkpt.pt
Done. Wrote output to out/ner_train_out/best/best_checkpt.txt
```

This creates a text file in the output directory which contains the path of the best model checkpoint.
```sh
$ ls out/classif_train_out/best
best_checkpt.txt

$ ls out/classif_train_out/best
best_checkpt.txt
```

### Model Evaluation

To estimate how the model will perform on the full dataset and in future runs, the best model is evaluated on the held-out test set. Since the model has not yet seen these data at all, it acts as a representative of new incoming data.

You can manually supply the path to the best model checkpoint as indicated in the above steps, or just `cat` the contents of the `best_checkpt.txt` file and pipe that into the evaluation command using `/dev/stdin` as shown below.

```sh
$ cat out/classif_train_out/best/best_checkpt.txt | \
  python3 src/class_final_eval.py \
    --out-dir out/classif_train_out/best/test_set_evaluation \
    --test-file out/classif_splits/test_paper_classif.csv \
    --checkpoint /dev/stdin
Done. Wrote output to out/classif_train_out/best/test_set_evaluation/.

$ cat out/ner_train_out/best/best_checkpt.txt | \
  python3 src/ner_final_eval.py \
    --out-dir out/ner_train_out/best/test_set_evaluation \
    --test-file out/ner_splits/test_ner.pkl \
    --checkpoint /dev/stdin
Done. Wrote output to out/ner_train_out/best/test_set_evaluation.

$ ls out/ner_train_out/best/test_set_evaluation
metrics.csv
```

### Performing Query

To get the full list of papers that we will assess, we can obtain the original query. Note that if papers are added retroactively, the yield of this query may change in the future, but should be largely the same.

```sh
$ python3 src/query_epmc.csv \
    --out-dir out/original_query \
    --from-date 2011 \
    --to-date 2021 \
    config/query.txt
Done. Wrote 2 files to out/original_query
```

2 Files are written to the output directory. One is the results of the query, the other is a text fle containing today's date

```sh
$ ls out/original_query
last_query_date.txt  query_results.csv
```

In order to always have the last query date text file in a known place, copy it over. That way, we can always pass in the file out/last_query_date/last_suery_date.txt when updating the inventory.
```sh
$ cp out/original_query/last_query_date.txt out/last_query_date/
```

### Predicting on Full Query Results

Now, we have the best trained models, and an indication of how they will perform on new data, so we can run them on the original full corpus.

First, run classification
```sh
$ cat out/classif_train_out/best/best_checkpt.txt | \
  python3 src/class_predict.py \
    --out-dir out/original_query/classification \
    --input-file out/original_query/query_results.csv \
    --checkpoint /dev/stdin

$ ls out/original_query/classification
predictions.csv
```

Filter to include only those papers predicted to describe biodata resources. This can be done with `grep -v` to get lines not containing the negative label.
```sh
$ grep -v 'not-bio-resource' \
    out/original_query/classification/predictions.csv \
    > out/original_query/classification/predicted_positives.csv
```

Run NER on the predicted positives
```sh
$ cat out/ner_train_out/best/best_checkpt.txt | \
  python3 src/ner_predict.py \
    --out-dir out/original_query/ner \
    --input-file out/original_query/classification/predicted_positives.csv \
    --checkpoint /dev/stdin

$ ls out/original_query/ner
predictions.csv
```

Extract URLs
```sh
$ python3 src/url_extractor.py \
    --out-dir out/original_query/url_predictions \
    out/original_query/ner/predictions.csv
```

Process names
```sh
$ python3 src/process_names.py \
    --out-dir out/original_query/processed_names \
    out/original_query/url_predictions/predictions.csv
```

Initial Deduplication
```sh
$ python3 src/initial_deduplicate.py \
    --out-dir out/original_query/initial_deduplication \
    out/original_query/processed_names/predictions.csv
```

Flag for selective manual review
```sh
$ python3 src/flag_for_review.py \
    --out-dir out/original_query/manual_review \
    --min-prob 0.978 \
    out/original_query/initial_deduplication/predictions.csv
```

Make directory for manually reviewed inventory
```sh
$ mkdir -p out/original_query/manually_reviewed
```

At this point, the inventory must be manually reviewed following the instructions.

Once it has been reviewed, place the file (`predictions.csv`) in the folder created above.

Process manually reviewed inventory
```sh
$ python3 src/process_manual_review.py \
    --out-dir out/original_query/processed_manual_review \
    out/original_query/manually_reviewed/predictions.csv
```

Check URL HTTP statuses
```sh
$ python3 src/check_urls.py \
    --out-dir out/original_query/url_checks \
    --chunk-size 200 \
    --num-tries 3 \
    --backoff 0.5 \
    out/original_query/processed_manual_review/predictions.csv
```

If the above gets interrupted, the partially completed output can be supplied as input to resume where it left off
```sh
$ python3 src/check_urls.py \
    --out-dir out/original_query/url_checks \
    --chunk-size 200 \
    --num-tries 3 \
    --backoff 0.5 \
    --partial out/original_query/url_checks/predictions.csv \
    out/original_query/processed_manual_review/predictions.csv
```

Get additional metadata from EuropePMC
```sh
$ python3 src/get_meta.py \
    --out-dir out/original_query/epmc_meta \
    --chunk-size 20 \
    out/original_query/url_checks/predictions.csv
```

Process country information
```sh
$ python3 src/process_countries.py \
    --out-dir out/original_query/processed_countries \
    out/original_query/epmc_meta/predictions.csv
```

## Updating the Inventory

These commands do not have to be run manually, since there are Snakemake pipeline and notebooks as described in [../README.md](../README.md). This example workflow is provided as additional documentation, and may be useful in debugging.

### Query EuropePMC

If this is the first time updating the inventory, the `--from-date` must be supplied manually. Here, I will use the last date from the original inventory. Otherwise, you can use the file resulting from the last run.
```sh
$ python3 src/query_epmc.py
    --out-dir out/new_query \
    --from-date out/last_query_date/last_query_date.txt \
    config/query.txt
Done. Wrote 2 files to out/new_query

$ cp out/new_query/last_query_date.txt out/last_query_date/
```

Two files are output to `--out-dir`: `last_query_date.txt` and `new_query_results.csv`. The former is then used in the next query.

### Obtain models

If the best trained models are not present in `out/classif_train_out/best/` and `out/ner_train_out/best/`, then they can be downloaded using the following command.

```sh
# command to get models
```

### Perform predictions and get other information

Classify the new results:
```sh
$ cat out/classif_train_out/best/best_checkpt.txt | \
  python3 src/class_predict.py \
    --out-dir out/new_query/classification \
    --input-file out/new_query/new_query_results.csv \
    --checkpoint /dev/stdin
```

Filter to include only those papers predicted to describe biodata resources.
```sh
$ grep -v 'not-bio-resource' \
    out/new_query/classification/predictions.csv \
    > out/new_query/classification/predicted_positives.csv
```

Run NER on the predicted positives
```sh
$ cat out/ner_train_out/best/best_checkpt.txt | \
  python3 src/ner_predict.py \
    --out-dir out/new_query/ner \
    --input-file out/new_query/classification/predicted_positives.csv \
    --checkpoint /dev/stdin
```

Extract URLs
```sh
$ python3 src/url_extractor.py \
    --out-dir out/new_query/urls \
    out/new_query/ner/predictions.csv
```

Check URLs
```sh
$ python3 src/check_urls.py \
    --out-dir out/new_query/check_urls \
    out/new_query/urls/predictions.csv
```

Get other metadata from EuropePMC query
```sh
$ python3 src/get_meta.py \
    --out-dir out/new_query/meta \
    out/new_query/check_urls/predictions.csv
```

