# inventory_2022
Public repository for the biodata resource inventory performed in 2022.

## I. Paper Clasification
###### 
**Task**: Predict if a paper is about a bio-data resource or not

<!-- ### Training Data -->

### Training

#### Usage
1. ``` pip install -r requierments.txt```
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
| predictive-field | Field in the dataframes to use for prediction. Can be one of ['title', 'abstract', 'title-abstract'] | 'title'| 
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

## II. NER Model
######
