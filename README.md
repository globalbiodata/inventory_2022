# inventory_2022 (Work in Progress)
Public repository for the biodata resource inventory performed in 2022.

## I. Paper Clasification
###### 
**Task**: Predict if a paper is about a bio-data resource or not

<!-- ### Training Data -->

### Training

#### Usage
1. ``` pip install -r requirements.txt```
2. For sanity checking: <br>
``` python train.py --num-training 100 --num-epochs 1 --sanity-check ```

3. To run on the entire data: <br>
``` python train.py --train-file 'path_to_train_data' --val-file 'path_to_val_data' ``` <br>

if --train-file and --val-file are not provided, the model assumes they are under ```data/train.csv``` and ```data/val.csv```
There are a number of other parameters that can be passed as command line arguments. The list of all arguments is:

| argument | usage | default_value |
| :- | :- | :-|
| train-file | Location of training file |'data/train.csv' | 
| val-file | Location of validation file | 'data/val.csv'| 
| test-file | Location of test file |'data/test.csv' | 
| model-name | Name of model to try. Can be one of: ['bert', 'biobert', 'scibert', 'pubmedbert', 'pubmedbert_pmc', 'bluebert', 'bluebert_mimic3', 'sapbert', 'sapbert_mean_token', 'bioelectra', 'bioelectra_pmc', 'electramed', 'biomed_roberta', 'biomed_roberta_chemprot', 'biomed_roberta_rct_500'] | 'scibert'| 
| predictive-field | Field in the dataframes to use for prediction. Can be one of ['title', 'abstract', 'title_abstract'] | 'title'| 
| labels-field | Field in the dataframes corresponding to the scores (0, 1) | 'curation_score'| 
| descriptive-labels | Descriptive labels corresponding to the [0, 1] numeric scores |['not-bio-resource', 'bio-resource'] | 
| sanity-check | True for sanity-check. Runs training on a smaller subset of the entire training data. | False | 
| num-training | Number of data points to run training on. If -1, training is ran an all the data. Useful for debugging. | -1 | 
| output-dir | Default directory to output checkpt and plot losses |'output_dir/' | 
| num-epochs | Number of Epochs | 10 | 
| batch-size | Batch Size | 32 | 
| max-len | Max Sequence Length | 256 | 
| learning-rate | Learning Rate |2e-5| 
| weight-decay | Weight Decay for Learning Rate | 0.0 | 
| lr-scheduler | True if using a Learning Rate Scheduler. More info here: https://huggingface.co/docs/transformers/main_classes/optimizer_schedules | False| 

After training, a checkpoint will be saved under ```output_dir```.
<!-- #### Experiments -->

#### Hyperparameters

|model_name| huggingface_model_version | learning_rate | batch_size | weight_decay| lr_scheduler | 
| :--- | :--- | :---: | :---: | :---: | :---: |
|bert_uncased|'bert_base_uncased'|3e-5|16|0|False|
|biobert|'dmis-lab/biobert-v1.1'|3e-5|32|0|False|
|scibert|'allenai/scibert_scivocab_uncased'|3e-5|-|0|False|
|pubmedbert|'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'|3e-5|32|0|True|
|pubmedbert_fulltext|'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'|3e-5|32|0|True|
|sapbert|'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'|2e-5|32|0.01|False|
|sapbert_mean_token|'cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token'|2e-5|32|0.01|False|
|bluebert|'bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12'|3e-5|32|0|True|
|bluebert_mimic3|'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12'|3e-5|32|0|False|
|electramed|'giacomomiolo/electramed_base_scivocab_1M'|5e-5|32|0|True|
|bioelectra|'kamalkraj/bioelectra-base-discriminator-pubmed'|5e-5|32|0|True|
|bioelectra_pmc|'kamalkraj/bioelectra-base-discriminator-pubmed-pmc'|5e-5|32|0|True|
|biomed_roberta|'allenai/biomed_roberta_base'|2e-5|16|0|False|
|biomed_roberta_chemprot|'allenai/dsp_roberta_base_dapt_biomed_tapt_chemprot_4169'|2e-5|16|0|False|
|biomed_roberta_rct_500|'allenai/dsp_roberta_base_dapt_biomed_tapt_rct_500'|2e-5|16|0|False|

### Prediction
Example: ``` python predict.py --input-file data/val.csv``` <br>
Requires a checkpoint from a trained model to be under ```output_dir/checkpoint``` <br>
One checkpoint from a model trained on BiomedRoberta can be downloaded from here: [Google Drive Link](https://drive.google.com/file/d/1qYDZvkpYqDWSIBLZg8J8x-Yzv4Fvuom0/view?usp=sharing)<br>
Metrics for the checkpoint above: ```F1: 0.9489051095	Precision: 0.9701492537	Recall: 0.9285714286```


## II. NER Model
###### 

## DATA DICTIONARIES

### Manual Curation

#### Variables for manual curation CSV files, e.g. manual_checks_all_YYYY-MM-DD.csv:

* **id** = article id
* **title** = article title
* **abstract** = article abstract
* **checked_by** = curator initials
* **kes_check** = kes determination where 0 = not an article describing data resource OR 1 = an article describing data resource
* **hji_check** = hji determination where 0 = not an article describing data resource OR 1 = an article describing data resource
* **curation_sum** = sum of curator values (iii_checks)
* **number_of_checks** = number of checks (by different curators)
* **curation_score** = curation_sum/number_of_checks (gives a "confidence score"" as done in Wren 2017); note that value other than 0 or 1 indiciate lack of agreement between curators
* **kes_notes** = raw notes documented by kes
* **hji_notes** = raw notes documented by hji

#### (likely temporary until we update the above file) For new variables within check classification conflicts CSV files, e.g. manual_checks_conflicts_2022-02-25.csv:

**FYI** 44/172 easily harmonized

* **index** = added index based manual_checks_all_2022-02-15.csv so can keep order as needed (esp needed if any chance of duplication of "id")
* **next steps** = "discuss" = discuss with kes, "kes review" = suggest for review (these seem like potentially clear/clearish mistakes; may be more based on the discussion for those that are less clear), or blank = no step (plan to merge as is into manual_checks_all_YYYY-MM-DD.csv - either now harmonized or so tricky that it should probably stay at 0.5 - that is, we shouldn't try to force agreement, could as Michaela's team to take a look though)

### Element Extraction

#### Variables for manual extraction of named entities *e.g.* extracted_elements_2022-02-15. This set only includes articles which had a **curation_score** of 1 as defined above.

* **id** = article id
* **title** = article title. Adjacent articles were not included (*e.g.* "Protein Ensemble Database" not "The Protein Ensemble Database").
* **abstract** = article abstract
* **name** = resource name
* **acronym** = resource acronym or shortened name, as presented in the title or abstract. This is sometimes the same as **name**.
* **url** = resource URL. Note, other URL's may have been present that were not that of the resource. These extraneous URL's were not extracted into this column.
* **short_description** = short description of the resource, as found in the abstract or title

**Notes**:

Version numbers were generally not included in **name** or **acronym** if there was white space between the element and version number (*e.g.* "CTDB" was recorded for "CTDB (v2.0)" while version number in "PDB-2-PBv3.0" was kept).

Many articles had several of the above elements. This could be for a few reasons:

* Multpile versions of an element, for instance when there are different **short_description**s in the title and abstract.
* Differences in case (*e.g.* "Human transporter database" vs "Human Transporter Database"). These are equivalent when case-insensitive, but case is deliberate in many titles.



In those cases, there will be multiple rows for the same article. For this reason, it would be best to either nest those fields (columns) with multiple entries, or to select columns of interest serially, and deduplicate.

In R this may look something like:

```
all_elements <- read.csv('extracted_elements_2022-02-15.csv')

names <- all_elements %>%
    select(id, title, abstract, name) %>%
    unique()

acronyms <- all_elements %>%
    select(id, title, abstract, acronym) %>%
    unique()
```
