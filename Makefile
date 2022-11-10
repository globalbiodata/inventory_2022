.PHONY: dryrun, setup, test, train_and_predict, update_inventory

dryrun_reproduction:
	snakemake \
	-s snakemake/train_predict.smk -np \
	--configfile config/train_predict.yml

setup:
	pip install -r requirements.txt
	echo "import nltk \nnltk.download('punkt')" | python3 /dev/stdin

test:
	python3 -m pytest -v \
	--flake8 --mypy --pylint  \
	--pylint-rcfile=config/.pylintrc  \
	tests/ \
	src/inventory_utils/*.py \
	src/*.py \

train_and_predict:
	snakemake \
	-s snakemake/train_predict.smk \
	--configfile config/train_predict.yml -c1

update_inventory:
	snakemake \
	-s snakemake/update_inventory.smk \
	--configfile config/update_inventory.yml -c1
