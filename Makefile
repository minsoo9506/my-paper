SHELL := /bin/bash

init:
	pip install -U pip
	pip install -r requirements.txt

format:
	black src/
	isort src/ --profile black