import pandas as pd
model_df = pd.read_table(config["models"]).set_index("model", drop=True)
model_df = model_df.fillna('')

rule all:
    input:
        config["classif_train_outdir"] + "/combined/best_checkpt.pt",
        config["classif_train_outdir"] + "/combined/combined_stats.csv",


rule split_classif_data:
    input:
        config["classif_data"],
    output:
        config["classif_splits_dir"] + "/train_paper_classif.csv",
        config["classif_splits_dir"] + "/val_paper_classif.csv",
        config["classif_splits_dir"] + "/test_paper_classif.csv",
    params:
        out_dir=config["classif_splits_dir"],
    shell:
        """
        python3 class_data_generator.py \
            -o {params.out_dir} \
            -r \
            {input}
        """


rule train_classif:
    input:
        train=config["classif_splits_dir"] + "/train_paper_classif.csv",
        val=config["classif_splits_dir"] + "/val_paper_classif.csv",
    output:
        config["classif_train_outdir"] + "/{model}/checkpt.pt",
        config["classif_train_outdir"] + "/{model}/train_stats.csv",
    params:
        out_dir=config["classif_train_outdir"],
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
            -m {params.hf_model} \
            -ne {params.epochs} \
            -t {input.train} \
            -v {input.val} \
            -o {params.out_dir} \
            -batch {params.batch_size} \
            -rate {params.learn_rate} \
            -decay {params.weight_decay} \
            {params.scheduler_flag}
        )2> {log}
        """


rule find_best_classifier:
    input:
        expand(
            "{d}/{model}/train_stats.csv",
            d=config["classif_train_outdir"],
            model=model_df.index,
        ),
    output:
        config["classif_train_outdir"] + "/combined/best_checkpt.pt",
        config["classif_train_outdir"] + "/combined/combined_stats.csv",
    params:
        out_dir=config["classif_train_outdir"] + "/combined",
    shell:
        """
        python3 model_picker.py \
            -o {params.out_dir} \
            {input}
        """


rule split_ner_data:
    input:
        config["ner_data"],
    output:
        config["ner_splits_dir"] + "/train_ner.csv",
        config["ner_splits_dir"] + "/val_ner.csv",
        config["ner_splits_dir"] + "/test_ner.csv",
    params:
        out_dir=config["ner_splits_dir"],
    shell:
        """
        python3 ner_data_generator.py \
            -o {params.out_dir} \
            -r \
            {input}
        """
