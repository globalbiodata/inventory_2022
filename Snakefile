rule all:
    input:
        expand(
            "{d}/checkpt_{m}_{e}_epochs",
            d=config["checkpoint_dir"],
            m=config["models"],
            e=config["epochs"],
        ),


rule train:
    input:
        train=config["train_data"],
        test=config["test_data"],
        val=config["val_data"],
    output:
        config["checkpoint_dir"] + "/checkpt_{model}_{epochs}_epochs",
    params:
        env=config["project_env"],
        out_dir=config["checkpoint_dir"],
        time=config["train_time"],
    threads: config["train_threads"]
    log:
        config["logk_dir"] + "/{model}_{epochs}.log",
    benchmark:
        config["benchmark_dir"] + "/{model}_{epochs}.txt"
    shell:
        """
        source activate {params.env}
        (./train.py \
            -m {wildcards.model} \
            -ne {wildcards.epochs} \
            -t {input.train} \
            -v {input.val} \
            -s {input.test} \
            -o {params.out_dir} \
        )2> {log}
        """
