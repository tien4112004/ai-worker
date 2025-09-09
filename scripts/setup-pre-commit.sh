echo "Installing pre-commit hooks..."

pip install pre-commit
pre-commit install
pre-commit clean
