.PHONY: test, dryrun

test:
	python3 -m pytest -v --flake8 --pylint --pylint-rcfile=config/.pylintrc --mypy \
	tests/ \
	src/class/train.py \
	src/class/predict.py \
	src/class/data_handler.py \
	src/class/utils.py

dryrun:
	snakemake -np --configfile config/config.yml