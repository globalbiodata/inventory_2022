import pandas as pd


include: "shared_rules.smk"


# Import tab separated file containing the configurations
# used for training each model.
model_df = pd.read_table(config["models"]).set_index("model", drop=True)
model_df = model_df.fillna("")


rule all:
    input:
        config["for_manual_review_dir"] + "/predictions.csv",
        config["classif_train_outdir"] + "/combined_train_stats/combined_stats.csv",
        config["classif_train_outdir"] + "/combined_test_stats/combined_stats.csv",
        config["ner_train_outdir"] + "/combined_train_stats/combined_stats.csv",
        config["ner_train_outdir"] + "/combined_test_stats/combined_stats.csv",


rule all_analysis:
    input:
        config["processed_countries"] + "/predictions.csv",
        config["figures_dir"] + "/class_val_set_performances.svg",
        config["figures_dir"] + "/class_val_set_performances.png",
        config["figures_dir"] + "/ner_val_set_performances.svg",
        config["figures_dir"] + "/ner_val_set_performances.png",
        config["figures_dir"] + "/combined_classification_table.docx",
        config["figures_dir"] + "/combined_ner_table.docx",
        config["figures_dir"] + "/ip_coordinates.png",
        config["figures_dir"] + "/ip_countries.png",
        config["figures_dir"] + "/author_countries.png",
        config["analysis_dir"] + "/analysed_metadata.txt",
        config["analysis_dir"] + "/inventory_re3data_fairsharing_summary.csv",
        config["analysis_dir"] + "/venn_diagram_sets.csv",
        config["figures_dir"] + "/text_mining_potential.csv",
        config["figures_dir"] + "/text_mining_potential_plot.png",
        config["figures_dir"] + "/text_mining_potential_plot.svg",
        config["figures_dir"] + "/inventory_funders.csv",
        config["figures_dir"] + "/funders_geo_counts.csv",
        config["figures_dir"] + "/funder_countries.png",


# Run EruopePMC query
rule query_epmc:
    output:
        query_results=config["query_out_dir"] + "/query_results.csv",
        date_file1=config["query_out_dir"] + "/last_query_date.txt",
        date_file2=config["last_date_dir"] + "/last_query_date.txt",
    params:
        out_dir=config["query_out_dir"],
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

        cp {output.date_file1} {output.date_file2}
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


# Combine training stats of all models
rule combine_classifier_stats:
    input:
        expand(
            "{d}/{model}/train_stats.csv",
            d=config["classif_train_outdir"],
            model=model_df.index,
        ),
    output:
        config["classif_train_outdir"] + "/combined_train_stats/combined_stats.csv",
    params:
        out_dir=config["classif_train_outdir"] + "/combined_train_stats",
    shell:
        """
        python3 src/combine_stats.py \
            -o {params.out_dir} \
            {input}
        """


# Select best trained classifier based on validation set
rule find_best_classifier:
    input:
        expand(
            "{d}/{model}/checkpt.pt",
            d=config["classif_train_outdir"],
            model=model_df.index,
        ),
    output:
        config["classif_train_outdir"] + "/best/best_checkpt.txt",
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


# Evaluate classification models on test set
rule evaluate_classifiers_on_test_set:
    input:
        infile=config["classif_splits_dir"] + "/test_paper_classif.csv",
        model=config["classif_train_outdir"] + "/{model}/checkpt.pt",
    output:
        config["classif_train_outdir"] + "/{model}/test_set_evaluation/metrics.csv",
    params:
        outdir=config["classif_train_outdir"] + "/{model}/test_set_evaluation",
    shell:
        """
        python3 src/class_final_eval.py \
            -o {params.outdir} \
            -t {input.infile} \
            -c {input.model}
        """


# Combine training stats of all article classification models on test set
rule combine_classifier_test_stats:
    input:
        expand(
            "{d}/{model}/test_set_evaluation/metrics.csv",
            d=config["classif_train_outdir"],
            model=model_df.index,
        ),
    output:
        config["classif_train_outdir"] + "/combined_test_stats/combined_stats.csv",
    params:
        out_dir=config["classif_train_outdir"] + "/combined_test_stats",
    shell:
        """
        python3 src/combine_stats.py \
            -o {params.out_dir} \
            {input}
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


# Combine training stats of all NER models
rule combine_ner_stats:
    input:
        expand(
            "{d}/{model}/train_stats.csv",
            d=config["ner_train_outdir"],
            model=model_df.index,
        ),
    output:
        config["ner_train_outdir"] + "/combined_train_stats/combined_stats.csv",
    params:
        out_dir=config["ner_train_outdir"] + "/combined_train_stats",
    shell:
        """
        python3 src/combine_stats.py \
            -o {params.out_dir} \
            {input}
        """


# Select best NER model based on validation set
rule find_best_ner:
    input:
        expand(
            "{d}/{model}/checkpt.pt",
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


# Evaluate NER models on test set
rule evaluate_ner_on_test_set:
    input:
        infile=config["ner_splits_dir"] + "/test_ner.pkl",
        model=config["ner_train_outdir"] + "/{model}/checkpt.pt",
    output:
        config["ner_train_outdir"] + "/{model}/test_set_evaluation/metrics.csv",
    params:
        outdir=config["ner_train_outdir"] + "/{model}/test_set_evaluation",
    shell:
        """
        python3 src/ner_final_eval.py \
            -o {params.outdir} \
            -t {input.infile} \
            -c {input.model}
        """


# Combine stats of all NER models on test set
rule combine_ner_test_stats:
    input:
        expand(
            "{d}/{model}/test_set_evaluation/metrics.csv",
            d=config["ner_train_outdir"],
            model=model_df.index,
        ),
    output:
        config["ner_train_outdir"] + "/combined_test_stats/combined_stats.csv",
    params:
        out_dir=config["ner_train_outdir"] + "/combined_test_stats",
    shell:
        """
        python3 src/combine_stats.py \
            -o {params.out_dir} \
            {input}
        """


# Perform deduplication on exact match names and URLs
rule initial_deduplication:
    input:
        config["processed_names_dir"] + "/predictions.csv",
    output:
        config["initial_dedupe_dir"] + "/predictions.csv",
    params:
        out_dir=config["initial_dedupe_dir"],
    shell:
        """
        python3 src/initial_deduplicate.py \
            -o {params.out_dir} \
            {input}
        """


# Create model metric plots and tables
rule analyze_performance_metrics:
    input:
        class_train=config["classification_train_stats"],
        class_test=config["classification_test_stats"],
        ner_train=config["ner_train_stats"],
        ner_test=config["ner_test_stats"],
    output:
        config["figures_dir"] + "/class_val_set_performances.svg",
        config["figures_dir"] + "/class_val_set_performances.png",
        config["figures_dir"] + "/ner_val_set_performances.svg",
        config["figures_dir"] + "/ner_val_set_performances.png",
        config["figures_dir"] + "/combined_classification_table.docx",
        config["figures_dir"] + "/combined_ner_table.docx",
    params:
        out_dir=config["figures_dir"],
    shell:
        """
        Rscript analysis/performance_metrics.R \
            -o {params.out_dir} \
            -cv {input.class_train} \
            -ct {input.class_test} \
            -nv {input.ner_train} \
            -nt {input.ner_test}
        """


# Create location data figures
rule process_location_data:
    input:
        config["final_inventory_file"],
    output:
        config["figures_dir"] + "/ip_coordinates.png",
        config["figures_dir"] + "/ip_countries.png",
        config["figures_dir"] + "/author_countries.png",
    params:
        out_dir=config["figures_dir"],
    shell:
        """
        Rscript analysis/location_information.R \
            -o {params.out_dir} \
            {input}
        """


# Analyse inventory metadata
rule process_metadata:
    input:
        config["final_inventory_file"],
    output:
        config["analysis_dir"] + "/analysed_metadata.txt",
    shell:
        """
        Rscript analysis/metadata_analysis.R \
            {input} \
            > {output}
        """


# Compare against re3data and FAIRsharing
rule compare_repositories:
    input:
        inventory=config["final_inventory_file"],
    output:
        config["analysis_dir"] + "/inventory_re3data_fairsharing_summary.csv",
        config["analysis_dir"] + "/venn_diagram_sets.csv",
    params:
        out_dir=config["analysis_dir"],
        login=config["fair_login_file"],
    shell:
        """
        Rscript analysis/comparison.R \
            -o {params.out_dir} \
            -c {params.login} \
            {input.inventory}
        """


# Gather and analyze additional metadata from EuropePMC
rule analyze_text_mining_potential:
    input:
        inventory=config["final_inventory_file"],
    output:
        config["figures_dir"] + "/text_mining_potential.csv",
        config["figures_dir"] + "/text_mining_potential_plot.png",
        config["figures_dir"] + "/text_mining_potential_plot.svg",
    params:
        out_dir=config["figures_dir"],
        query=config["query_string"],
    shell:
        """
        Rscript analysis/epmc_metadata.R \
            -o {params.out_dir} \
            -q {params.query} \
            {input.inventory}
        """


# Gather and analyze funder data
rule analyze_funding_agencies:
    input:
        inventory=config["final_inventory_file"],
    output:
        config["figures_dir"] + "/inventory_funders.csv",
    params:
        config["figures_dir"],
    shell:
        """
        Rscript analysis/funders.R \
            -o {params.out_dir} \
            {input.inventory}
        """


# Gather and analyze funder data
rule analyze_funding_countries:
    input:
        config["curated_funders"],
    output:
        config["figures_dir"] + "/funders_geo_counts.csv",
        config["figures_dir"] + "/funder_countries.png",
    params:
        config["figures_dir"],
    shell:
        """
        Rscript analysis/funders.R \
            -o {params.out_dir} \
            {input}
        """
