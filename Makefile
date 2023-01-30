.PHONY: dryrun, setup, test, train_and_predict, update_inventory

dryrun_reproduction:
	snakemake \
	-s snakemake/train_predict.smk -np \
	--configfile config/train_predict.yml

setup:
	pip install -r requirements.txt
	echo "import nltk \nnltk.download('punkt')" | python3 /dev/stdin
	pip install --upgrade numpy
	Rscript -e 'install.packages("renv")'
	Rscript -e 'renv::restore()'

setup_for_updating:
	pip install -r requirements.txt
	echo "import nltk \nnltk.download('punkt')" | python3 /dev/stdin
	pip install --upgrade numpy

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

process_manually_reviewed_original:
# process manually reviewed
	snakemake \
	-s snakemake/train_predict.smk \
	--configfile config/train_predict.yml \
	-c 1 \
	--until process_countries
# analyze performance metrics
	snakemake \
	-s snakemake/train_predict.smk \
	--configfile config/train_predict.yml \
	-c 1 \
	--until analyze_performance_metrics
# process location data
	snakemake \
	-s snakemake/train_predict.smk \
	--configfile config/train_predict.yml \
	-c 1 \
	--until process_location_data
# process metadata
	snakemake \
	-s snakemake/train_predict.smk \
	--configfile config/train_predict.yml \
	-c 1 \
	--until process_metadata
# Compare repositories
	snakemake \
	-s snakemake/train_predict.smk \
	--configfile config/train_predict.yml \
	-c 1 \
	--until compare_repositories
# Text mining potential
	snakemake \
	-s snakemake/train_predict.smk \
	--configfile config/train_predict.yml \
	-c 1 \
	--until analyze_text_mining_potential

update_inventory:
	snakemake \
	-s snakemake/update_inventory.smk \
	--configfile config/update_inventory.yml -c1

process_manually_reviewed_update:
# process manually reviewed
	snakemake \
	-s snakemake/update_inventory.smk \
	--configfile config/update_inventory.yml \
	-c 1 \
	--until process_countries