.PHONY: test, dryrun

test:
	python3 -m pytest -v --flake8 --pylint tests/ train.py

dryrun:
	snakemake -np --configfile config/config.yml