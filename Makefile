.PHONY: test, dryrun

test:
	python3 -m pytest -v --flake8 --pylint --pylint-rcfile=config/.pylintrc --mypy \
	tests/ \
	src/train.py \
	src/predict.py \
	src/data_handler.py \
	src/utils.py

dryrun:
	snakemake -np --configfile config/config.yml