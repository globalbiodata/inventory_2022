# Find best classifier model name based on a file glob
best_classifier_pattern = config["best_classifier_dir"] + "/{model}/best_checkpt.pt"
(CLASSIFIER,) = glob_wildcards(best_classifier_pattern)

# Find best NER model name based on a file glob
best_ner_pattern = config["best_ner_dir"] + "/{model}/best_checkpt.pt"
(NER,) = glob_wildcards(best_ner_pattern)


rule all:
    input:
        "data/new_paper_predictions/urls/predictions.csv",


# Run EuropePMC query with new dates
rule query_epmc:
    input:
        from_date="data/last_query_date.txt",
        query=config["query_string"],
    output:
        "data/last_query_date.txt",
        "data/new_query_results.csv",
    params:
        out_dir="data",
    shell:
        """
        python3 src/query_epmc.py \
            -o {params.out_dir} \
            --from-date {input.from_date} \
            {input.query}
        """


# Perform classification of new query results
rule classify_papaers:
    input:
        infile="data/new_query_results.csv",
        classifier=expand(best_classifier_pattern, model=CLASSIFIER),
    output:
        "data/new_paper_predictions/classification/predictions.csv",
    params:
        out_dir="data/new_paper_predictions/classification",
    shell:
        """
        python3 src/class_predict.py \
            -o {params.out_dir} \
            -c {input.classifier} \
            -i {input.infile}
        """


# Filter out only predicted biodata resources
rule filter_positives:
    input:
        "data/new_paper_predictions/classification/predictions.csv",
    output:
        "data/new_paper_predictions/classification/predicted_positives.csv",
    shell:
        """
        grep -v 'not-bio-resource' {input} >> {output}
        """


# Predict NER on predicted biodata resource papers
rule ner_full_corpus:
    input:
        classifier=expand(best_ner_pattern, model=NER),
        infile="data/new_paper_predictions/classification/predicted_positives.csv",
    output:
        "data/new_paper_predictions/ner/predictions.csv",
    params:
        out_dir="data/new_paper_predictions/ner",
    shell:
        """
        python3 src/ner_predict.py \
            -o {params.out_dir} \
            -c {input.classifier} \
            -i {input.infile}
        """


# Extract out URLS
rule get_urls:
    input:
        "data/new_paper_predictions/ner/predictions.csv",
    output:
        "data/new_paper_predictions/urls/predictions.csv",
    params:
        out_dir="data/new_paper_predictions/urls",
    shell:
        """
        python3 src/url_extractor.py \
            -o {params.out_dir} \
            {input}
        """
