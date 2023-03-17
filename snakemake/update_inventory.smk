include: "shared_rules.smk"


rule all:
    input:
        config["for_manual_review_dir"] + "/predictions.csv",


# Run EuropePMC query with new dates
rule query_epmc:
    output:
        query_results=config["query_out_dir"] + "/query_results.csv",
        date_file1=config["query_out_dir"] + "/last_query_dates.txt",
        date_file2=config["last_date_dir"] + "/last_query_dates.txt",
    params:
        out_dir=config["query_out_dir"],
        query=config["query_string"],
        from_date=config["query_from_date"],
        to_date=config["query_to_date"],
    shell:
        """
        python3 src/query_epmc.py \
            -o {params.out_dir} \
            --from-date {params.from_date} \
            --to-date {params.to_date} \
            {params.query}

        cp {output.date_file1} {output.date_file2}
        """


# Perform deduplication on exact match names and URLs
rule initial_deduplication:
    input:
        new_file=config["processed_names_dir"] + "/predictions.csv",
        previous_file=config["previous_inventory"],
    output:
        config["initial_dedupe_dir"] + "/predictions.csv",
    params:
        out_dir=config["initial_dedupe_dir"],
    shell:
        """
        python3 src/initial_deduplicate.py \
            -o {params.out_dir} \
            -p {input.previous_file} \
            {input.new_file}
        """
