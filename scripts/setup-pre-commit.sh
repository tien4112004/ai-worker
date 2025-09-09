echo "Installing pre-commit hooks..."

pip install pre-commit
pre-commit install
pre-commit clean

echo "Running initial pre-commit check..."
pre-commit run --all-files
