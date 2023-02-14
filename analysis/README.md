# Data Analysis

This directory contains R scripts for some analysis of the inventory conducted in 2022. They are stored here rather than [src](../src/) since their reuse is likely limited and is strictly related to analysis. However, these scripts are used in the [train and predict Snakemake pipeline](../snakemake/train_predict.smk).

```sh
.
├── location_information.R   # Generate maps of resource location metadata
├── metadata_analysis.R      # Perform high-level metadata analysis
└── performance_metrics.R    # Create plots and tables of model performances
└── epmc_metadata.R          # Retrieve ePMC metadata to determine OA, full text, etc.
└── comparison.R             # Retrieve life sci resources from FAIRsharing and re3data
└── funders.R                # Analyse funder metadata by article and biodata resource
└── funders_geo.R            # Analyse top 200 funders by country
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

## `epmc_metadata.R`

The final inventory file is supplied as input and the Europe PMC API is queried to determined if the article has a CC license, is open access, has full text available, has text mined term, and has text mined accession numbers. Note that all but full text are found by querying the PMIDs found in the final inventory file; for full text, the original query was restricted to return only those as OA and having full text availability for the entire corpus and then those PMIDs were matched against the PMIDs found in the final inventory.

1 file is output:
* `text_mining_potential.csv`: A summary table of article counts (Y (Yes) or N (No)) and percentages

## `comparison.R`

Inputs are retrieved by querying the records available from the re3data.org API and the FAIRsharing API. Returns are filtered to life science resources and then compared resources identified in the final inventory. 

n files are output:

tbd: Venn Diagram?

## `funders.R`

The final inventory file is supplied as input and the Europe PMC API is queried to retrieve "agency" metadata from individual articles (note that biodata resources in the inventory have concatenated "grantID" and "agency" values for resources with >1 article). This scripts retrieves "agency" for each article, when present, to analyze the supporting funding organizations identified.

1 file is output:
* `inventory_funders_2023-01-20.csv`: Deduplicated funder names with total unique article count, total unique biodata resource count, associated article PMIDs (list) and associated biodata resources (list).

## `funders_geo.R`

The output file from funders.R (inventory_funders_2023-01-20.csv) was manually curated to determine countries for funders mentioned >2 times and mapped to ISO.3166-1.alpha-3 country codes. The resulting file, funders_geo_200.csv, is used as the input for this script which groups by unique country to get summary statistics. Note that for agency names, there is some ambiguity via either unclear parent-child relationships (e.g. NIH vs. NIGMS) or inconsistent naming (e.g. National Key Research and Development Program vs. National Key Research Program of China).

1 file is output:
* `funders_geo_counts_2023-02-10.csv`: By country summary with count unique agency names, count unique biodata resources, agency names (list) and biodata resource names (list).


