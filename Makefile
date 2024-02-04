.PHONY: dryrun, setup, test, train_and_predict, update_inventory

dryrun_reproduction:
	snakemake \
	-s snakemake/train_predict.smk -np \
	--configfile config/train_predict.yml

setup:
	python3.8 -m pip install -r requirements.txt
	echo "import nltk \nnltk.download('punkt')" | python3 /dev/stdin
	pip install --upgrade numpy
	Rscript -e 'install.packages("renv"), repos="http://cran.us.r-project.org"'
	Rscript -e 'renv::restore()'

setup_for_updating:
	python3.8 -m pip install -r requirements.txt
	echo "import nltk \nnltk.download('punkt')" | python3 /dev/stdin
	pip install --upgrade numpy==1.23

test:
	python3.8 -m pytest -v \
	--flake8 --mypy --pylint  \
	--pylint-rcfile=config/.pylintrc  \
	src/inventory_utils/*.py \
	src/*.py \

train_and_predict:
	snakemake \
	-s snakemake/train_predict.smk \
	--configfile config/train_predict.yml -c1

process_manually_reviewed_original:
	snakemake \
	-s snakemake/train_predict.smk \
	--configfile config/train_predict.yml \
	-c 1 \
	--until all_analysis

update_inventory:
	snakemake \
	-s snakemake/update_inventory.smk \
	--configfile config/update_inventory.yml -c1

process_manually_reviewed_update:
	snakemake \
	-s snakemake/update_inventory.smk \
	--configfile config/update_inventory.yml \
	-c 1 \
	--until process_countries