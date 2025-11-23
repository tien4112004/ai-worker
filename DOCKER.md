# Docker Deployment Guide

This guide covers how to build, run, and deploy the AI Worker application using Docker.

## Prerequisites

- Docker Engine 20.10+ or Docker Desktop
- Docker Compose 2.0+
- (Optional) Docker Buildx for multi-platform builds

## Quick Start

### Using Docker Compose (Recommended)

1. **Development Environment:**
   ```bash
   # Start the application
   docker-compose up -d

   # View logs
   docker-compose logs -f

   # Stop the application
   docker-compose down
   ```

2. **Production Environment:**
   ```bash
   # Start with production configuration
   docker-compose -f docker-compose.prod.yml up -d

   # View logs
   docker-compose -f docker-compose.prod.yml logs -f
   ```

### Using Docker CLI

1. **Build the image:**
   ```bash
   docker build -t ai-worker:latest .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     --name ai-worker \
     -p 8081:8081 \
     --env-file .env \
     ai-worker:latest
   ```

3. **View logs:**
   ```bash
   docker logs -f ai-worker
   ```

4. **Stop the container:**
   ```bash
   docker stop ai-worker
   docker rm ai-worker
   ```

## Using Makefile Commands

We provide convenient Makefile commands for common Docker operations:

```bash
# Build Docker image
make docker-build

# Build with BuildKit optimization
make docker-build-fast

# Run container
make docker-run

# Run container in foreground (see logs)
make docker-run-fg

# Stop container
make docker-stop

# View logs
make docker-logs

# Start with docker-compose
make docker-compose-up

# Stop docker-compose
make docker-compose-down

# Test Docker image
make docker-test

# Clean Docker resources
make docker-clean
```

## Environment Variables

The application requires the following environment variables:

### Required API Keys
- `GOOGLE_API_KEY` - Google Generative AI API key
- `OPENAI_API_KEY` - OpenAI API key (optional)
- `ANTHROPIC_API_KEY` - Anthropic API key (optional)
- `OPENROUTER_API_KEY` - OpenRouter API key (optional)
- `DEEPSEEK_API_KEY` - DeepSeek API key (optional)

### Application Configuration
- `APP_NAME` - Application name (default: "ai-worker")
- `DEFAULT_MODEL` - Default LLM model (default: "gemini-2.5-flash-lite")
- `LLM_TEMPERATURE` - Temperature for text generation (default: 0.7)
- `LLM_MAX_TOKENS` - Maximum tokens for generation (default: 2048)
- `MAX_RETRIES` - Maximum retry attempts (default: 3)
- `LOG_LEVEL` - Logging level (default: "info")

### CORS Configuration
- `ALLOWED_ORIGINS` - Allowed origins for CORS (default: "*")
- `ALLOWED_CREDENTIALS` - Allow credentials (default: "true")
- `ALLOWED_METHODS` - Allowed HTTP methods (default: "*")
- `ALLOWED_HEADERS` - Allowed headers (default: "*")

### Vertex AI Configuration (Optional)
- `VERTEX_PROJECT_ID` - GCP project ID for Vertex AI
- `VERTEX_LOCATION` - GCP location for Vertex AI

## Docker Compose Files

### `docker-compose.yml` (Development)
- Includes volume mounts for live code reloading
- Suitable for local development
- Uses bridge network

### `docker-compose.prod.yml` (Production)
- Uses pre-built image from registry
- Optimized for production use
- Includes resource limits
- Enhanced logging configuration

## Multi-Platform Builds

To build for multiple platforms (amd64 and arm64):

```bash
# Create a new builder
docker buildx create --use --name multiplatform-builder

# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t ghcr.io/tien4112004/ai-worker:latest \
  --push \
  .
```

Or use the Makefile:
```bash
make docker-build-multiplatform
```

## GitHub Container Registry

### Pulling Images

```bash
# Pull latest image
docker pull ghcr.io/tien4112004/ai-worker:latest

# Pull specific version
docker pull ghcr.io/tien4112004/ai-worker:v1.0.0
```

### Using in Docker Compose

Update your `docker-compose.yml`:
```yaml
services:
  ai-worker:
    image: ghcr.io/tien4112004/ai-worker:latest
    # ... rest of configuration
```

## CI/CD Pipeline

The project includes a GitHub Actions workflow (`.github/workflows/docker.yml`) that:

1. **Builds** the Docker image on push to `main` or `develop`
2. **Pushes** to GitHub Container Registry
3. **Tags** images based on:
   - Branch name (e.g., `main`, `develop`)
   - Git SHA (e.g., `main-abc1234`)
   - Semantic versions (e.g., `v1.0.0`, `1.0`, `1`)
   - `latest` tag for main branch
4. **Scans** for security vulnerabilities using Trivy
5. **Tests** the built image to ensure it starts correctly

### Triggering Builds

- **Push to main/develop:** Automatic build and push
- **Create tag:** `git tag v1.0.0 && git push origin v1.0.0`
- **Pull Request:** Build only (no push)

## Health Checks

The container includes health checks at:
- **Endpoint:** `http://localhost:8081/docs`
- **Interval:** Every 30 seconds
- **Timeout:** 10 seconds
- **Retries:** 3 attempts
- **Start Period:** 40 seconds

Check health status:
```bash
docker inspect --format='{{.State.Health.Status}}' ai-worker
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker logs ai-worker

# Check if port is already in use
lsof -i :8081

# Run in foreground to see immediate output
make docker-run-fg
```

### Health check failing
```bash
# Check health status
docker inspect ai-worker | jq '.[0].State.Health'

# Test endpoint manually
curl http://localhost:8081/docs

# Increase start period in docker-compose.yml
```

### Permission issues
```bash
# Rebuild without cache
docker-compose build --no-cache

# Check file permissions
ls -la app/
```

### Memory issues
```bash
# Increase memory limit in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G
```

## Security Best Practices

1. **Don't commit secrets:** Use environment variables or secrets management
2. **Use non-root user:** The container runs as `appuser` (non-root)
3. **Scan for vulnerabilities:** Regular security scans with Trivy
4. **Keep images updated:** Regularly update base images
5. **Minimal attack surface:** Multi-stage builds reduce image size

## Resource Management

Default resource limits:
- **CPU:** 1-2 cores
- **Memory:** 1-2 GB

Adjust in `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 4G
    reservations:
      cpus: '2'
      memory: 2G
```

## Monitoring

### View real-time stats
```bash
docker stats ai-worker
```

### Export logs
```bash
docker logs ai-worker > ai-worker.log
```

### Integration with monitoring tools
- **Prometheus:** Expose metrics endpoint
- **Grafana:** Create dashboards
- **ELK Stack:** Centralized logging

## Production Deployment

For production deployments, consider:

1. **Use orchestration:** Kubernetes, Docker Swarm, or ECS
2. **Load balancing:** Nginx, HAProxy, or cloud load balancers
3. **Auto-scaling:** Based on CPU/memory usage
4. **Secret management:** AWS Secrets Manager, HashiCorp Vault
5. **Monitoring:** Prometheus + Grafana, DataDog, New Relic
6. **Logging:** Centralized logging with ELK or CloudWatch
7. **Backups:** Regular backups of volumes and data

## Support

For issues or questions:
- Check logs: `docker logs ai-worker`
- Review health status: `docker inspect ai-worker`
- Open an issue on GitHub
