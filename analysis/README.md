# Data Analysis

This directory contains R scripts for some analysis of the inventory conducted in 2022. They are stored here rather than [src](../src/) since their reuse is likely limited and is strictly related to analysis. However, these scripts are used in the [train and predict Snakemake pipeline](../snakemake/train_predict.smk).

```sh
.
├── location_information.R   # Generate maps of resource location metadata
├── metadata_analysis.R      # Perform high-level metadata analysis
└── performance_metrics.R    # Create plots and tables of model performances
```

All R scripts are command-line executable and take output files from the inventory as inputs for analysis. Usage statements are available through the `-h|--help` flag.

## `location_information.R`

The final inventory file is supplied as input, output directory is specified with `-o|--out-dir`, and 3 maps are generated:

* `ip_coordinates.png`: IP host coordinates dot plot
* `ip_countries.png`: IP host countries heatmap, with country fill color scaled to country name count
* `author_countries.png`: Author affiliation countries heatmap, with country fill color scaled to country name count

## `metadata_analysis.R`

The final inventory file is supplied as input, and various metadata statistics are output to stdout. To easily save the output of this, simply redirect (`>`) the output to a file. For example, running from the root of the repository:

```sh
$ Rscript analysis/metadata_analysis.R \
    data/final_inventory_2022.csv \
    > analysis/analysed_metadata.txt
```

In this case, no output will be seen in the terminal, but the output will be present in `analysis/analysed_metadata.txt`.

Information included in this analysis:

* Number of unique articles
* Number of resources with at least 1 URL returning 2XX or 3XX
* Number of resources with at least 1 WayBack URL
* Number of resources with grant agency data

## `performance_metrics.R`

This script conducts analysis on the model performance metrics on the validation and test sets., Output directory is specified with `-o|--out-dir`. Four files are needed as input:

* `-cv|--class-train`: Classification training and validation set statistics
* `-ct|--class-test`: Classification test set statistics
* `-nv|--ner-train`: NER training and validation set statistics
* `-nt|--ner-test`: NER test set statistics

The defaults for these arguments are the files stored in the repository, which is the results of the inventory conducted in 2022.

Six files are output:

* `class_val_set_performance.svg` and `class_val_set_performance.png`: Bar chart showing the performance of all article classification models on the validation set. Metrics include *F*1-score, precision, and recall. Models are in decreasing order of precision.
* `ner_val_set_performance.svg` and `ner_val_set_performance.png`: Bar chart showing the performance of all NER models on the validation set. Metrics include *F*1-score, precision, and recall. Models are in decreasing order of *F*1-score.
* `combined_classification_table.docx`: A Microsoft Word doc with a table showing the performance of all article classification models on the validation and test sets. Models are in decreasing order of precision on the validation set.
* `combined_ner_table.docx`: A Microsoft Word doc with a table showing the performance of all NER models on the validation and test sets. Models are in decreasing order of *F*1-score on the validation set.
