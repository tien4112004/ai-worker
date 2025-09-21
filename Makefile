.PHONY: setup install-dev compile-deps upgrade-deps sync-deps run test test-with-coverage build clean

# Development setup
setup: compile-deps sync-deps
	cp .env.sample .env
	bash scripts/setup-pre-commit.sh

# Compile requirements files from .in files
compile-deps:
	pip-compile requirements.in
	pip-compile requirements-dev.in
	pip-compile requirements-test.in

# Upgrade all dependencies to latest versions
upgrade-deps:
	pip-compile --upgrade requirements.in
	pip-compile --upgrade requirements-dev.in
	pip-compile --upgrade requirements-test.in

# Sync virtual environment with compiled requirements
sync-deps:
	pip-sync requirements.txt requirements-dev.txt

# Install development dependencies
install-dev:
	pip install -r requirements-dev.txt

# Install test dependencies
install-test:
	pip install -r requirements-test.txt

# Run the application
run:
	python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8080

# Run tests
test:
	pytest

# Run tests with coverage
test-with-coverage:
	pytest --cov=app --cov-report=html --cov-report=term

# Build Docker image
build:
	bash scripts/build-image.sh

# Clean generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/

# Update pip-tools itself
update-pip-tools:
	pip install --upgrade pip-tools

# Show outdated packages
show-outdated:
	pip list --outdated
