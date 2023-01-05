# Overview

This directory contains the data used for model training, testing, and validation as well as several final output files.

```sh
.
├── classif_metrics/               # Article classification model performance metrics
|   ├── combined_train_stats.csv   # Performance on training and validation sets
|   └── combined_train_stats.csv   # Performance on witheld test set
├── ner_metrics/                   # NER model performance metrics
|   ├── combined_train_stats.csv   # Performance on training and validation sets
|   └── combined_train_stats.csv   # Performance on witheld test set
├── epmc_query_results_2022.csv    # EuropePMC query return used in 2022 inventory
├── final_inventory_2022.csv       # Final inventorry generated in 2022
├── manual_classifications.csv     # Manual article classifications
└── manual_ner_extraction.csv      # Manual NER extraction
```
## `*/combined_train_stats.csv`

The two files `classif_metrics/combined_train_stats.csv` and `ner_metrics/combined_train_stats.csv` have the same columns. They contain obtained during fine-tuning of the article classification and NER models, respectively.

### `classif_metrics/combined_train_stats.csv`

Metrics of each model were output by [src/class_train.py](../src/class_train.py), and combined by [src/combine_stats.py](../src/combine_stats.py) to generate this file.

### `ner_metrics/combined_train_stats.csv`

Metrics of each model were output by [src/ner_train.py](../src/ner_train.py), and combined by [src/combine_stats.py](../src/combine_stats.py) to generate this file.

### Columns of both files:

* **epoch**: Training epoch, beginning at 0 and going to 9 (10 epochs)
* **train_precision**: Precision on training set
* **train_recall**: Recall on training set
* **train_f1**: *F*1-score on training set
* **train_loss**: Loss on training set
* **val_precision**: Precision on validation set
* **val_recall**: Recall on validation set
* **val_f1**: *F*1-score on validation set
* **val_loss**: Loss on validation set
* **model_name**: Model name (corresponding to the **model** column of [config/models_info.tsv](../config/models_info.tsv))

## `*/combined_test_stats`

The two files `classif_metrics/combined_test_stats.csv` and `ner_metrics/combined_test_stats.csv` have the same columns. They contain obtained during evaluation on the witheld test set of the article classification and NER models, respectively.

### `classif_metrics/combined_test_stats.csv`

Metrics of each model were output by [src/class_final_eval.py](../src/class_final_eval.py), and combined by [src/combine_stats.py](../src/combine_stats.py) to generate this file.

### `ner_metrics/combined_test_stats.csv`

Metrics of each model were output by [src/ner_final_eval.py](../src/ner_final_eval.py), and combined by [src/combine_stats.py](../src/combine_stats.py) to generate this file.

### Columns of both files:

* **model**: Model name (corresponding to the **model** column of [config/models_info.tsv](../config/models_info.tsv))
* **precision**: Precision on test set
* **recall**: Recall on test set
* **f1**: *F*1-score on test set
* **loss**: Loss on test set

## `epmc_query_results_2022.csv`

EuropePMC query results that were used for generation of the inventory in 2022. The following columns are included:

* **id**: article id
* **title**: article title
* **abstract**: article abstract
* **publication_date**: article first publication date

## `final_inventory_2022.csv`

The final output file of the inventory conducted in 2022.

Information on the contents of this file are available in the [main README](../README.md#final-inventory-output).

## `manual_classifications.csv`

This file contains the manual classifications of 1634 articles by kes (Kenneth Schackart) and hji (Heidi Imker). This set was split into training, validation, and testing splits for fine-tuning and evaluation of the article classification models.

* **id**: article id
* **title**: article title
* **abstract**: article abstract
* **checked_by**: curator initials
* **kes_check**: kes determination where 0 = not an article describing data resource OR 1 = an article describing data resource
* **hji_check**: hji determination where 0 = not an article describing data resource OR 1 = an article describing data resource
* **curation_sum**: sum of curator values (iii_checks)
* **number_of_checks**: number of checks (by different curators)
* **curation_score**: curation_sum/number_of_checks (gives a "confidence score"" as done in Wren 2017); note that value other than 0 or 1 indiciate lack of agreement between curators
* **kes_notes**: raw notes documented by kes
* **hji_notes**: raw notes documented by hji

## `manual_ner_extraction.csv`

The file contains the manual NER extraction from articles manually classifiedf to describe a biodata resource. Curation was performed by Kenneth Schackart, and validated by Heidi Imker.

* **id**: article id
* **title**: article title. Adjacent articles were not included (*e.g.* "Protein Ensemble Database" not "The Protein Ensemble Database").
* **abstract**: article abstract
* **name**: resource name
* **acronym**: resource acronym or shortened name, as presented in the title or abstract. This is sometimes the same as **name**.
* **url**: resource URL. Note, other URL's may have been present that were not that of the resource. These extraneous URL's were not extracted into this column.
* **short_description**: short description of the resource, as found in the abstract or title

**Notes**:

Version numbers were generally not included in **name** or **acronym** if there was white space between the element and version number (*e.g.* "CTDB" was recorded for "CTDB (v2.0)" while version number in "PDB-2-PBv3.0" was kept).

Many articles had several of the above elements. This could be for a few reasons:

* Multiple versions of an element, for instance when there are different **short_description**s in the title and abstract.
* Differences in case (*e.g.* "Human transporter database" vs "Human Transporter Database"). These are equivalent when case-insensitive, but case is deliberate in many titles.
