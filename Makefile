.PHONY: setup install-dev compile-deps upgrade-deps sync-deps run test test-with-coverage build clean

venv:
	python -m venv venv
	source venv/bin/activate
	pip install --upgrade pip

# Development setup
setup:
	pip install pip-tools
	make compile-deps
	make sync-deps
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
	python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8081

# Run the application with default port
run-default:
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

# ==================== Docker Commands ====================

# Build Docker image locally
docker-build:
	docker build -t ai-worker:latest .

# Build Docker image with BuildKit optimizations
docker-build-fast:
	DOCKER_BUILDKIT=1 docker build -t ai-worker:latest .

# Build multi-platform Docker image
docker-build-multiplatform:
	docker buildx build --platform linux/amd64,linux/arm64 -t ai-worker:latest .

# Run Docker container
docker-run:
	docker run -d \
		--name ai-worker \
		-p 8081:8081 \
		--env-file .env \
		ai-worker:latest

# Run Docker container in foreground (with logs)
docker-run-fg:
	docker run --rm \
		--name ai-worker \
		-p 8081:8081 \
		--env-file .env \
		ai-worker:latest

# Stop Docker container
docker-stop:
	docker stop ai-worker || true
	docker rm ai-worker || true

# View Docker container logs
docker-logs:
	docker logs -f ai-worker

# Execute shell in running container
docker-shell:
	docker exec -it ai-worker /bin/bash

# Start with docker-compose (development)
docker-compose-up:
	docker-compose up -d

# Start with docker-compose (production)
docker-compose-up-prod:
	docker-compose -f docker-compose.prod.yml up -d

# Stop docker-compose
docker-compose-down:
	docker-compose down

# View docker-compose logs
docker-compose-logs:
	docker-compose logs -f

# Rebuild and restart docker-compose
docker-compose-rebuild:
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d

# Clean Docker resources
docker-clean:
	docker-compose down -v
	docker rmi ai-worker:latest || true
	docker system prune -f

# Test Docker image
docker-test:
	@echo "Building Docker image..."
	@docker build -t ai-worker:test .
	@echo "Starting container..."
	@docker run -d --name ai-worker-test -p 8082:8081 --env-file .env ai-worker:test
	@echo "Waiting for container to be healthy..."
	@sleep 10
	@echo "Testing health endpoint..."
	@curl -f http://localhost:8082/docs || (docker logs ai-worker-test && exit 1)
	@echo "Stopping test container..."
	@docker stop ai-worker-test
	@docker rm ai-worker-test
	@docker rmi ai-worker:test
	@echo "Docker test passed!"

# Push Docker image to registry (GitHub Container Registry)
docker-push-ghcr:
	docker tag ai-worker:latest ghcr.io/tien4112004/ai-worker:latest
	docker push ghcr.io/tien4112004/ai-worker:latest

# Pull Docker image from registry
docker-pull-ghcr:
	docker pull ghcr.io/tien4112004/ai-worker:latest
	docker tag ghcr.io/tien4112004/ai-worker:latest ai-worker:latest
