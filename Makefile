.PHONY: test, dryrun

test:
	python3 -m pytest -v --flake8 --pylint --pylint-rcfile=.pylintrc \
	tests/ \
	src/train.py

dryrun:
	snakemake -np --configfile config/config.yml