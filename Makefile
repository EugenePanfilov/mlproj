.PHONY: install train predict monitor test

install:        ## pip install -r requirements.txt
	pip install -r requirements.txt

train:          ## python train.py --config configs/config.yaml
	python train.py --config configs/config.yaml --run-name default

predict:        ## python predict.py --model-dir artifacts/LATEST --input-csv data/sample.csv --output-csv out.csv
	python predict.py --model-dir artifacts/LATEST --input-csv data/sample.csv --output-csv out.csv

monitor:        ## python monitor_run.py --model-dir artifacts/LATEST --reference-csv data/ref.csv --current-csv data/curr.csv
	python monitor_run.py --model-dir artifacts/LATEST --reference-csv data/ref.csv --current-csv data/curr.csv

test:           ## pytest -q
	pytest -q

clean-artifacts: ## remove all artifact runs
	rm -rf artifacts/*