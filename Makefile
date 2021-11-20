SHELL := /bin/bash

init:
	pip install -U pip
	pip install -r requirements.txt

format:
	black src/

lint:
	pytest src/ --flake8