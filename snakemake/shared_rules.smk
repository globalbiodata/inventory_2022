# Predict classification of entire corpus
rule classify_papers:
    input:
        model=config["classif_train_outdir"] + "/best/best_checkpt.txt",
        infile=config["query_out_dir"] + "/query_results.csv",
    output:
        config["classif_out_dir"] + "/predictions.csv",
    params:
        out_dir=config["classif_out_dir"],
    shell:
        """
        python3 src/class_predict.py \
            -o {params.out_dir} \
            -i {input.infile} \
            -c "$(< {input.model})"
        """


# Filter out only predicted biodata resources
rule filter_positives:
    input:
        config["classif_out_dir"] + "/predictions.csv",
    output:
        config["classif_out_dir"] + "/predicted_positives.csv",
    shell:
        """
        grep -v 'not-bio-resource' {input} > {output}
        """


# Predict NER on predicted biodata resource papers
rule ner_predict:
    input:
        infile=config["classif_out_dir"] + "/predicted_positives.csv",
        model=config["ner_train_outdir"] + "/best/best_checkpt.txt",
    output:
        config["ner_out_dir"] + "/predictions.csv",
    params:
        out_dir=config["ner_out_dir"],
    shell:
        """
        python3 src/ner_predict.py \
            -o {params.out_dir} \
            -i {input.infile} \
            -c "$(< {input.model})"
        """


# Extract URLs from title and abstract
rule extract_urls:
    input:
        config["ner_out_dir"] + "/predictions.csv",
    output:
        config["extract_url_dir"] + "/predictions.csv",
    params:
        out_dir=config["extract_url_dir"],
    shell:
        """
        python3 src/url_extractor.py \
            -o {params.out_dir} \
            {input}
        """


# Check URL status, get locations, and check for Wayback Snapshot
rule check_urls:
    input:
        config["extract_url_dir"] + "/predictions.csv",
    output:
        config["check_url_dir"] + "/predictions.csv",
    params:
        out_dir=config["check_url_dir"],
    shell:
        """
        python3 src/check_urls.py \
            -o {params.out_dir} \
            {input}
        """


# Filter the inventory
rule filter_results:
    input:
        config["check_url_dir"] + "/predictions.csv",
    output:
        config["filtered_results_dir"] + "/predictions.csv",
    params:
        out_dir=config["filtered_results_dir"],
        min_urls=config["min_urls"],
        max_urls=config["max_urls"],
        min_prob=config["manual_review_prob"],
    log:
        config["filtered_results_dir"] + "/log.txt",
    shell:
        """
        (python3 src/filter_results \
            -nu {params.min_urls} \
            -xu {params.max_urls} \
            -np {params.min_prob} \
            -o {params.out_dir} \
            {input}) 2>&1 | tee {log}
        """


# Deduplicate the inventory
rule deduplicat_results:
    input:
        config["filtered_results_dir"] + "/predictions.csv",
    output:
        config["deduplicated_dir"] + "/predictions.csv",
    params:
        out_dir=config["deduplicated_dir"],
    log:
        config["deduplicated_dir"] + "/log.txt",
    shell:
        """
        (python3 src/deduplicate.py \
            -o {params.out_dir} \
            {input}) 2>&1 | tee {log}
        """
