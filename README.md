# ai-worker

## Requirements
- Python 3.10+
- pip (or pip3)

## Setup
- Prepare virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

- Install dependencies
```bash
git clone
cd ai-worker
pip install -r requirements.txt
```

- Prepare environment variables
```bash
cp .env.sample .env
# Edit .env file to add your API keys and configurations
```

## How to run
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

## Development

### Pre-commit Hooks
Set up pre-commit hooks for code quality:
```bash
bash scripts/setup-pre-commit.sh
```

### Testing
Run tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=app --cov-report=html
```

### Docker

#### Quick Build & Run
Use the automated build script:
```bash
# Build with default settings
bash scripts/build-image.sh

# Build with custom tag
bash scripts/build-image.sh ai-worker:v1.0.0

# Build and push to registry
bash scripts/build-image.sh ai-worker:latest true
```
#### Manual Docker Commands
Build and run manually:
```bash
docker build -t ai-worker .
docker run -p 8080:8080 --env-file .env ai-worker
```

#### Docker Compose
For development with services:
```bash
# Start all services
docker-compose up --build

# Start in background
docker-compose up -d --build

# View logs
docker-compose logs -f ai-worker

# Stop services
docker-compose down
```

#### Docker Image Features
- **Multi-stage build** for optimized image size
- **Non-root user** for security
- **Health checks** for container monitoring
- **Build metadata** (git commit, build date)
- **Automatic testing** during build process

## Scripts

The project includes several utility scripts in the `scripts/` directory:

### Build Scripts
- **`scripts/build-image.sh`**: Automated Docker image build script
  - Builds Docker image with proper tagging
  - Runs automated tests on the built image
  - Supports pushing to container registry
  - Includes git metadata in build

### Development Scripts
- **`scripts/setup-pre-commit.sh`**: Sets up pre-commit hooks for code quality

### Usage Examples
```bash
# Build Docker image
bash scripts/build-image.sh

# Build with specific tag
bash scripts/build-image.sh my-app:v1.2.3

# Build and push to registry
bash scripts/build-image.sh my-app:latest true

# Setup development environment
bash scripts/setup-pre-commit.sh
```

## CI/CD

This project includes comprehensive CI/CD pipelines:

### GitHub Actions Workflows

1. **CI Pipeline** (`.github/workflows/ci.yml`):
   - Tests across Python 3.10, 3.11, 3.12
   - Code linting (black, isort, flake8, mypy)
   - Security scanning (bandit)
   - Pre-commit validation
   - Application startup testing

2. **Docker Build** (`.github/workflows/docker.yml`):
   - Multi-platform Docker image building
   - Container registry publishing
   - Docker image testing

3. **Deployment** (`.github/workflows/deploy.yml`):
   - Staging deployment on main branch
   - Production deployment on version tags
   - Release creation

### Code Quality Tools

- **Black**: Python code formatting
- **isort**: Import sorting
- **flake8**: Code linting
- **mypy**: Static type checking
- **bandit**: Security vulnerability scanning
- **pre-commit**: Git hooks for code quality

### Environment Setup

The CI pipeline automatically:
- Installs dependencies
- Sets up test environment
- Runs comprehensive test suite
- Generates coverage reports
- Validates code quality
- Tests application startup

## Test with curl

```bash
curl -X POST "http://localhost:8080/api/v1/outline/generate"
    -H "Content-Type: application/json"
    -d '{
    "topic": "Introduction to Tắt đèn of Ngô Tất Tố",
    "slide_count": 5,
    "audience": "university students",
    "model": "gemini-2.5-flash-lite",
    "learning_objective": "",
    "language": "vi",
    "targetAge": "5-10"
    }'
```

## Folder structure

```bash
.
├── app
│   ├── api
│   │   ├── endpoints
│   │   │   ├── generate.py
│   │   │   └── __pycache__
│   │   ├── __pycache__
│   │   │   └── router.cpython-310.pyc
│   │   └── router.py
│   ├── core
│   │   ├── config.py
│   │   └── depends.py
│   ├── llms
│   │   ├── factory.py
│   │   └── service.py
│   ├── main.py
│   ├── schemas
│   │   ├── image_content.py
│   │   └── slide_content.py
│   └── services
│       └── content_service.py
├── README.md
├── requirements.txt
└── tests
```
