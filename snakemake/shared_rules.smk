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
        cat {input.model} | \
        python3 src/class_predict.py \
            -o {params.out_dir} \
            -i {input.infile} \
            -c /dev/stdin
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
        cat  {input.model} | \
        python3 src/ner_predict.py \
            -o {params.out_dir} \
            -i {input.infile} \
            -c /dev/stdin
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
