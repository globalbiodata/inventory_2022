import pandas as pd

# Import tab separated file containing the configurations
# used for training each model.
model_df = pd.read_table(config["models"]).set_index("model", drop=True)
model_df = model_df.fillna("")


rule all:
    input:
        "data/full_corpus_predictions/predicted_positives.csv",
        "data/full_corpus_predictions/ner/predictions.csv",
        "data/full_corpus_predictions/urls/predictions.csv",
        config["classif_train_outdir"] + "/best/test_set_evaluation/metrics.csv",
        config["ner_train_outdir"] + "/best/test_set_evaluation/metrics.csv",


# Run EruopePMC query
rule query_epmc:
    output:
        query=config["full_corpus"],
        last_date="data/last_query_date.txt",
    params:
        out_dir="data",
        begin_date=config["initial_query_start"],
        end_date=config["initial_query_end"],
        query=config["query_string"],
    shell:
        """
        python3 src/query_epmc.py \
            -o {params.out_dir} \
            --from-date {params.begin_date} \
            --to-date {params.end_date} \
            {params.query}

        mv {params.out_dir}/new_query_results.csv {output.query}
        """


# Split curated classification set into train, val, and test
rule split_classif_data:
    input:
        config["classif_data"],
    output:
        config["classif_splits_dir"] + "/train_paper_classif.csv",
        config["classif_splits_dir"] + "/val_paper_classif.csv",
        config["classif_splits_dir"] + "/test_paper_classif.csv",
    params:
        out_dir=config["classif_splits_dir"],
        splits=config["split_ratios"],
    shell:
        """
        python3 src/class_data_generator.py \
            -o {params.out_dir} \
            --splits {params.splits} \
            -r \
            {input}
        """


# Train each classifier
rule train_classif:
    input:
        train=config["classif_splits_dir"] + "/train_paper_classif.csv",
        val=config["classif_splits_dir"] + "/val_paper_classif.csv",
    output:
        config["classif_train_outdir"] + "/{model}/checkpt.pt",
        config["classif_train_outdir"] + "/{model}/train_stats.csv",
    params:
        out_dir=config["classif_train_outdir"] + "/{model}",
        metric=config["class_criteria_metric"],
        epochs=config["classif_epochs"],
        hf_model=lambda w: model_df.loc[w.model, "hf_name"],
        batch_size=lambda w: model_df.loc[w.model, "batch_size"],
        learn_rate=lambda w: model_df.loc[w.model, "learning_rate"],
        weight_decay=lambda w: model_df.loc[w.model, "weight_decay"],
        scheduler_flag=lambda w: model_df.loc[w.model, "scheduler"],
    log:
        config["classif_log_dir"] + "/{model}.log",
    benchmark:
        config["classif_benchmark_dir"] + "/{model}.txt"
    shell:
        """
        (python3 src/class_train.py \
            -c {params.metric} \
            -m {params.hf_model} \
            -ne {params.epochs} \
            -t {input.train} \
            -v {input.val} \
            -o {params.out_dir} \
            -batch {params.batch_size} \
            -rate {params.learn_rate} \
            -decay {params.weight_decay} \
            -r \
            {params.scheduler_flag}
        )2> {log}
        """


# Select best trained classifier based on validation F1 score
rule find_best_classifier:
    input:
        expand(
            "{d}/{model}/checkpt.pt",
            d=config["classif_train_outdir"],
            model=model_df.index,
        ),
    output:
        onfig["classif_train_outdir"] + "/best/best_checkpt.txt",
    params:
        out_dir=config["classif_train_outdir"] + "/best",
        metric=config["class_criteria_metric"],
    shell:
        """
        python3 src/model_picker.py \
            -o {params.out_dir} \
            -m {params.metric} \
            {input}
        """


# Evaluate model on test set
rule evaluate_best_classifier:
    input:
        infile=config["classif_splits_dir"] + "/test_paper_classif.csv",
        model=config["classif_train_outdir"] + "/best/best_checkpt.txt",
    output:
        config["classif_train_outdir"] + "/best/test_set_evaluation/metrics.csv",
    params:
        outdir=config["classif_train_outdir"] + "/best/test_set_evaluation",
    shell:
        """
        cat {input.model} | \
        python3 src/class_final_eval.py \
            -o {params.outdir} \
            -t {input.infile} \
            -c /dev/stdin
        """


# Predict classification of entire corpus
rule classify_full_corpus:
    input:
        model=config["classif_train_outdir"] + "/best/best_checkpt.txt",
        infile=config["full_corpus"],
    output:
        "data/full_corpus_predictions/classification/predictions.csv",
    params:
        out_dir="data/full_corpus_predictions/classification",
    shell:
        """
        cat {input.model} | \
        python3 src/class_predict.py \
            -o {params.out_dir} \
            -i {input.infile} \
            -c /dev/stdin
        """


# Filter out only predicted biodata resources
rule filter_positives:
    input:
        "data/full_corpus_predictions/classification/predictions.csv",
    output:
        "data/full_corpus_predictions/classification/predicted_positives.csv",
    shell:
        """
        grep -v 'not-bio-resource' {input} > {output}
        """


# Split curated NER set into train, val, and test
rule split_ner_data:
    input:
        config["ner_data"],
    output:
        config["ner_splits_dir"] + "/train_ner.csv",
        config["ner_splits_dir"] + "/val_ner.csv",
        config["ner_splits_dir"] + "/test_ner.csv",
        config["ner_splits_dir"] + "/train_ner.pkl",
        config["ner_splits_dir"] + "/val_ner.pkl",
        config["ner_splits_dir"] + "/test_ner.pkl",
    params:
        out_dir=config["ner_splits_dir"],
        splits=config["split_ratios"],
    shell:
        """
        python3 src/ner_data_generator.py \
            -o {params.out_dir} \
            --splits {params.splits} \
            -r \
            {input}
        """


# Train each NER model
rule train_ner:
    input:
        train=config["ner_splits_dir"] + "/train_ner.pkl",
        val=config["ner_splits_dir"] + "/val_ner.pkl",
    output:
        config["ner_train_outdir"] + "/{model}/checkpt.pt",
        config["ner_train_outdir"] + "/{model}/train_stats.csv",
    params:
        out_dir=config["ner_train_outdir"] + "/{model}",
        metric=config["ner_criteria_metric"],
        epochs=config["ner_epochs"],
        hf_model=lambda w: model_df.loc[w.model, "hf_name"],
        batch_size=lambda w: model_df.loc[w.model, "batch_size"],
        learn_rate=lambda w: model_df.loc[w.model, "learning_rate"],
        weight_decay=lambda w: model_df.loc[w.model, "weight_decay"],
        scheduler_flag=lambda w: model_df.loc[w.model, "scheduler"],
    log:
        config["ner_log_dir"] + "/{model}.log",
    benchmark:
        config["ner_benchmark_dir"] + "/{model}.txt"
    shell:
        """
        (python3 src/ner_train.py \
            -c {params.metric} \
            -m {params.hf_model} \
            -ne {params.epochs} \
            -t {input.train} \
            -v {input.val} \
            -o {params.out_dir} \
            -batch {params.batch_size} \
            -rate {params.learn_rate} \
            -decay {params.weight_decay} \
            -r \
            {params.scheduler_flag}
        )2> {log}
        """


# Select best NER model based on validation F1 score
rule find_best_ner:
    input:
        expand(
            "{d}/{model}/train_stats.csv",
            d=config["ner_train_outdir"],
            model=model_df.index,
        ),
    output:
        config["ner_train_outdir"] + "/best/best_checkpt.txt",
    params:
        out_dir=config["ner_train_outdir"] + "/best",
        metric=config["ner_criteria_metric"],
    shell:
        """
        python3 src/model_picker.py \
            -o {params.out_dir} \
            -m {params.metric} \
            {input}
        """


# Evaluate model on test set
rule evaluate_best_ner:
    input:
        infile=config["ner_splits_dir"] + "/test_ner.pkl",
        model=config["ner_train_outdir"] + "/best/best_checkpt.txt",
    output:
        config["ner_train_outdir"] + "/best/test_set_evaluation/metrics.csv",
    params:
        outdir=config["ner_train_outdir"] + "/best/test_set_evaluation",
    shell:
        """
        cat {input.model} | \
        python3 src/ner_final_eval.py \
            -o {params.outdir} \
            -t {input.infile} \
            -c /dev/stdin
        """


# Predict NER on predicted biodata resource papers
rule ner_full_corpus:
    input:
        model=config["ner_train_outdir"] + "/best/best_checkpt.txt",
        infile="data/full_corpus_predictions/classification/predicted_positives.csv",
    output:
        "data/full_corpus_predictions/ner/predictions.csv",
    params:
        out_dir="data/full_corpus_predictions/ner",
    shell:
        """
        cat  {input.model} | \
        python3 src/ner_predict.py \
            -o {params.out_dir} \
            -i {input.infile} \
            -c /dev/stdin
        """


# Extract out URLS
rule get_urls:
    input:
        "data/full_corpus_predictions/ner/predictions.csv",
    output:
        "data/full_corpus_predictions/urls/predictions.csv",
    params:
        out_dir="data/full_corpus_predictions/urls",
    shell:
        """
        python3 src/url_extractor.py \
            -o {params.out_dir} \
            {input}
        """
