.PHONY: setup run test test-with-coverage build

setup:
	cp .env.sample .env
	bash scripts/setup-pre-commit.sh
	pip install -r requirements.txt

run:
	python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8080

test:
	pytest

test-with-coverage:
	pytest --cov=app --cov-report=html

build:
	bash scripts/build-image.sh
