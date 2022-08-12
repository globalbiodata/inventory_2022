.PHONY: dryrun, setup, test, train_and_predict, update_inventory

dryrun_reproduction:
	snakemake -s snakemake/train_predict.smk -np --configfile config/train_predict.yml

setup:
	pip install -r requirements.txt
	echo "import nltk \nnltk.download('punkt')" | python3 /dev/stdin

test:
	python3 -m pytest -v --flake8 --pylint --pylint-rcfile=config/.pylintrc --mypy \
	tests/ \
	src/inventory_utils/*.py \
	src/check_urls.py \
	src/class_data_generator.py \
	src/class_train.py \
	src/class_predict.py \
	src/ner_data_generator.py \
	src/ner_train.py \
	src/ner_predict.py \
	src/model_picker.py \
	src/class_final_eval.py \
	src/ner_final_eval.py \
	src/query_epmc.py \
	src/url_extractor.py \

train_and_predict:
	snakemake -s snakemake/train_predict.smk --configfile config/train_predict.yml -c1

update_inventory:
	snakemake -s snakemake/update_inventory.smk --configfile config/update_inventory.yml -c1