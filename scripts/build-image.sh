#!/bin/bash

# Build script for AI Worker Docker image
# Usage: bash scripts/build-image.sh [tag]

set -e

# Default values
DEFAULT_TAG="ai-worker:latest"
REGISTRY="ghcr.io"
REPOSITORY="ltt204/ai-worker"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
TAG=${1:-$DEFAULT_TAG}
PUSH=${2:-false}

# Get version from git if available
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    GIT_COMMIT=$(git rev-parse --short HEAD)
    GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    if [[ "$TAG" == *":latest" ]] && [[ "$GIT_BRANCH" != "main" ]]; then
        TAG="${TAG%:latest}:${GIT_BRANCH}"
        warn "Building on branch '$GIT_BRANCH', using tag: $TAG"
    fi
else
    GIT_COMMIT="unknown"
    GIT_BRANCH="unknown"
    BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
fi

log "Starting Docker image build..."
log "Tag: $TAG"
log "Git Commit: $GIT_COMMIT"
log "Git Branch: $GIT_BRANCH"
log "Build Date: $BUILD_DATE"

# Check if Dockerfile exists
if [[ ! -f "Dockerfile" ]]; then
    error "Dockerfile not found in current directory"
    exit 1
fi

# Check if requirements.txt exists
if [[ ! -f "requirements.txt" ]]; then
    error "requirements.txt not found in current directory"
    exit 1
fi

# Build the Docker image
log "Building Docker image with tag: $TAG"

docker build \
    --build-arg BUILD_DATE="$BUILD_DATE" \
    --build-arg GIT_COMMIT="$GIT_COMMIT" \
    --build-arg GIT_BRANCH="$GIT_BRANCH" \
    --tag "$TAG" \
    --progress=plain \
    .

if [[ $? -eq 0 ]]; then
    success "Docker image built successfully: $TAG"
else
    error "Docker image build failed"
    exit 1
fi

# Test the built image
log "Testing the built image..."
CONTAINER_NAME="ai-worker-test-$(date +%s)"

# Start container for testing
docker run -d \
    --name "$CONTAINER_NAME" \
    -p 8081:8080 \
    -e GOOGLE_API_KEY=test_key \
    -e OPENAI_API_KEY=test_key \
    -e ANTHROPIC_API_KEY=test_key \
    "$TAG"

if [[ $? -eq 0 ]]; then
    log "Container started successfully, waiting for health check..."

    # Wait for container to be healthy
    for i in {1..30}; do
        if docker exec "$CONTAINER_NAME" curl -f http://localhost:8080/docs > /dev/null 2>&1; then
            success "Container health check passed"
            break
        fi

        if [[ $i -eq 30 ]]; then
            error "Container health check failed"
            docker logs "$CONTAINER_NAME"
            docker stop "$CONTAINER_NAME" > /dev/null 2>&1
            docker rm "$CONTAINER_NAME" > /dev/null 2>&1
            exit 1
        fi

        sleep 2
    done

    # Cleanup test container
    docker stop "$CONTAINER_NAME" > /dev/null 2>&1
    docker rm "$CONTAINER_NAME" > /dev/null 2>&1
    success "Container test completed successfully"
else
    error "Failed to start test container"
    exit 1
fi

# Show image details
log "Image details:"
docker images "$TAG" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

# Optional: Push to registry
if [[ "$PUSH" == "true" || "$PUSH" == "yes" || "$PUSH" == "1" ]]; then
    REGISTRY_TAG="$REGISTRY/$REPOSITORY:$(echo $TAG | cut -d':' -f2)"

    log "Tagging image for registry: $REGISTRY_TAG"
    docker tag "$TAG" "$REGISTRY_TAG"

    log "Pushing image to registry..."
    docker push "$REGISTRY_TAG"

    if [[ $? -eq 0 ]]; then
        success "Image pushed successfully: $REGISTRY_TAG"
    else
        error "Failed to push image to registry"
        exit 1
    fi
fi

success "Build process completed successfully!"
log "You can now run the image with:"
log "  docker run -p 8080:8080 --env-file .env $TAG"
log ""
log "Or use docker-compose:"
log "  docker-compose up --build"
