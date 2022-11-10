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
        max_urls=config["max_urls"],
    shell:
        """
        python3 src/url_extractor.py \
            -o {params.out_dir} \
            -x {params.max_urls} \
            {input}
        """


# Process predcited resource names
rule process_names:
    input:
        config["extract_url_dir"] + "/predictions.csv",
    output:
        config["processed_names_dir"] + "/predictions.csv",
    params:
        out_dir=config["processed_names_dir"],
    shell:
        """
        python3 src/process_names.py \
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


# Flag rows for manual review
rule flag_for_review:
    input:
        config["initial_dedupe_dir"] + "/predictions.csv",
    output:
        config["for_manual_review_dir"] + "/predictions.csv",
    params:
        out_dir=config["for_manual_review_dir"],
        min_prob=config["min_best_name_prob"],
        new_dir=config["manually_reviewed_dir"],
    shell:
        """
        python3 src/flag_for_review.py \
            -o {params.out_dir} \
            -p {params.min_prob} \
            {input}

        mkdir -p {params.new_dir}
        echo "Inventory flagged for manual review."
        echo "Once manual review is finished place file in {params.new_dir}."
        """


# Process manual review
rule process_manual_review:
    input:
        config["manually_reviewed_dir"] + "/predictions.csv",
    output:
        config["processed_manual_review"] + "/predictions.csv",
    params:
        out_dir=config["processed_manual_review"],
    shell:
        """
        python3 src/process_manual_review.py \
            -o {params.out_dir} \
            {input}
        """


# Check URL http statuses
rule check_urls:
    input:
        config["processed_manual_review"] + "/predictions.csv",
    output:
        config["check_url_dir"] + "/predictions.csv",
    params:
        out_dir=config["check_url_dir"],
        chunk_size=config["chunk_size"],
        num_tries=config["num_tries"],
        backoff=config["backoff"],
    shell:
        """
        python3 src/check_urls.py \
            -s {params.chunk_size} \
            -n {params.num_tries} \
            -b {params.backoff} \
            -o {params.out_dir} \
            {input}
        """
