rule all:
    input:
        expand(
            "{d}/{model}_checkpt.pt",
            d=config["classif_train_outdir"],
            m=config["models"],
        ),


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
        config["classif_train_outdir"] + "/{model}_checkpt.pt",
        config["classif_train_outdir"] + "/{model}_train_stats.csv",
    params:
        out_dir=config["classif_train_outdir"],
        epochs=config["classif_epochs"],
    log:
        config["classif_log_dir"] + "/{model}.log",
    benchmark:
        config["classif_benchmark_dir"] + "/{model}.txt"
    shell:
        """
        (python3 src/class_train.py \
            -m {wildcards.model} \
            -ne {params.epochs} \
            -t {input.train} \
            -v {input.val} \
            -o {params.out_dir} \
        )2> {log}
        """


rule find_best_classifier:
    input:
        expand(
            "{d}/{model}_train_stats.csv",
            d=config["classif_train_outdir"],
            m=config["models"],
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
