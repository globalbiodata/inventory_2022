# Snakemake Pipelines

Snakemake is used to organize the various steps involved in reproducing the original results or updating the inventory by defining workflows. This allows for easy execution of many steps in a specified order to obtain the desired outputs.

This directory contains both workflow files, and a file of rules (steps) shared by multiple workflows.

```sh
.
├── README.md
├── shared_rules.smk      # File with shared rules
├── train_predict.smk     # Workflow for reproducing results
└── update_inventory.smk  # Workflow for updating inventory
```

These files should not need to be edited, since they just capture the workflow logic, while configurations are separate and present in [config/](../config/). 

However, if any of these files get edited, it can be nice to format them. This makes them have consistent formatting, and can spot some syntax errors:

```sh
# To format the Snakefiles
$ snakefmt *.smk
```


## `shared_rules.smk`

The rules in this file are imported for use in the other workflows by having the line `include: shared_rules.smk` in the workflows. Shared rules are modularized so as to adhere to D.R.Y. (Don't Repeat Yourself) as much as possible. These are mostly downstream steps, since those occur during both the original results and in updating the inventory.

Note that the rules in this file often specify values obtained from the config file. For instance:

```python
infile=config["query_out_dir"] + "/query_results.csv",
```

Each snakemake workflow will be utilizing a different config file. So the value of `config["query_out_dir"]` will vary based on what workflow is using the rule. That also means that the config file of each workflow using these rules must have the appropriate keys (*e.g.* "query_out_dir").

## `train_predict.smk`

This is the workflow used to obtain the results in the manuscript, and can be run to reproduce our results. Instructions on how to do so are present in the [root README.md](../README.md).

Configurations for this workflow are present in [train_predict.yml](../config/train_predict.yml) and [models_info.tsv](../config/models_info.tsv)

## `update_inventory.smk`

This workflow can be used update the inventory. Instructions on how to do so are present in the [root README.md](../README.md).

Configurations for this workflow are present in [update_inventory.yml](../config/update_inventory.yml).