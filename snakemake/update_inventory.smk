include: "shared_rules.smk"


rule all:
    input:
        config["extract_url_dir"] + "/predictions.csv",


# Run EuropePMC query with new dates
rule query_epmc:
    input:
        from_date=config["last_date_dir"] + "/last_query_date.txt",
    output:
        query_results=config["query_out_dir"] + "/query_results.csv",
        date_file1=config["query_out_dir"] + "/last_query_date.txt",
        date_file2=config["last_date_dir"] + "/last_query_date.txt",
    params:
        out_dir=config["query_out_dir"],
        query=config["query_string"],
    shell:
        """
        python3 src/query_epmc.py \
            -o {params.out_dir} \
            --from-date {input.from_date} \
            {input.query}

        cp {output.date_file1} {output.date_file2}
        """
