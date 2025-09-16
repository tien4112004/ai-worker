# ai-worker

## Requirements

- Python 3.10+
- pip (or pip3)

## Setup

### Prerequisites

- Python 3.10+
- pip (or pip3)
- pip-tools (for dependency management)

### Quick Setup

1. **Prepare virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

2. **Install pip-tools**

```bash
pip install --upgrade pip pip-tools
```

3. **Clone and setup project**

```bash
git clone <repository-url>
cd ai-worker
make setup  # This will compile dependencies and sync environment
```

4. **Prepare environment variables**

```bash
cp .env.sample .env
# Edit .env file to add your API keys and configurations
```

### VertexAI Configuration

For Google VertexAI integration, you need to set up additional environment variables and service account credentials:

1. **Set up service account**:
   - Create a service account in Google Cloud Console
   - Download the service account JSON key file
   - Place the file as `service-account.json` in the project root

2. **Export VertexAI environment variables**:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=./service-account.json
export VERTEX_PROJECT_ID=your-gcp-project-id
export VERTEX_LOCATION=us-central1
```

3. **Update .env file**:

Edit your `.env` file to include the VertexAI configuration:

```bash
# For Google VertexAI
GOOGLE_APPLICATION_CREDENTIALS=./service-account.json
VERTEX_PROJECT_ID=your-gcp-project-id
VERTEX_LOCATION=us-central1
```

**Note**: Make sure your Google Cloud project has the Vertex AI API enabled and your service account has the necessary permissions (Vertex AI User role).

### Manual Setup (Alternative)

If you prefer to install dependencies manually:

```bash
# Compile requirements from .in files
make compile-deps

# Install all dependencies for development
make sync-deps

# Or install specific dependency groups
pip install -r requirements.txt              # Production dependencies
pip install -r requirements-dev.txt          # Development dependencies
pip install -r requirements-test.txt         # Test dependencies
```

## How to run
****
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

## Development

### Dependency Management with pip-tools

This project uses [pip-tools](https://pip-tools.readthedocs.io/) for reproducible dependency management. Dependencies are defined in `.in` files and compiled to `.txt` files.

#### Dependency Files Structure

- `requirements.in` - Production dependencies
- `requirements-dev.in` - Development dependencies (includes production deps)
- `requirements-test.in` - Test dependencies (includes production deps)
- `requirements.txt` - Compiled production dependencies (auto-generated)
- `requirements-dev.txt` - Compiled development dependencies (auto-generated)
- `requirements-test.txt` - Compiled test dependencies (auto-generated)

#### Common Dependency Tasks

**Add a new dependency:**
```bash
# Add to appropriate .in file (e.g., requirements.in for production)
echo "new-package>=1.0.0" >> requirements.in
make compile-deps  # Compile all .in files
make sync-deps     # Sync environment
```

**Update all dependencies:**
```bash
make upgrade-deps  # Upgrade to latest compatible versions
make sync-deps     # Sync environment with new versions
```

**Update specific dependency:**
```bash
pip-compile --upgrade-package package-name requirements.in
make sync-deps
```

**Sync environment with requirements:**
```bash
make sync-deps  # Ensures your environment matches compiled requirements exactly
```

**Install development environment:**
```bash
make install-dev  # Install dev dependencies only
```

**Show outdated packages:**
```bash
make show-outdated
```

#### Makefile Targets

- `make setup` - Complete project setup (compile deps + sync + setup pre-commit)
- `make compile-deps` - Compile all .in files to .txt files
- `make upgrade-deps` - Upgrade all dependencies to latest versions
- `make sync-deps` - Sync virtual environment with compiled requirements
- `make install-dev` - Install development dependencies
- `make install-test` - Install test dependencies
- `make update-pip-tools` - Update pip-tools itself

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
- **pip-tools integration** for reproducible builds
- **Non-root user** for security
- **Health checks** for container monitoring
- **Build metadata** (git commit, build date)
- **Automatic testing** during build process

#### Docker and pip-tools

The Docker build process uses the compiled `requirements.txt` files for reproducible container builds:

```dockerfile
# Production image uses compiled requirements
COPY requirements.txt .
RUN pip install --no-deps -r requirements.txt

# Development image can use dev requirements
COPY requirements-dev.txt .
RUN pip install --no-deps -r requirements-dev.txt
```

Make sure to compile your requirements before building Docker images:

```bash
make compile-deps  # Ensure requirements.txt is up to date
bash scripts/build-image.sh
```

## Scripts

The project includes several utility scripts and Makefile targets for common development tasks:

### Makefile Targets

**Dependency Management:**
```bash
make setup           # Complete project setup
make compile-deps    # Compile .in files to .txt files
make upgrade-deps    # Upgrade all dependencies
make sync-deps       # Sync environment with requirements
make install-dev     # Install development dependencies
make install-test    # Install test dependencies
make show-outdated   # Show outdated packages
make update-pip-tools # Update pip-tools itself
```

**Application:**
```bash
make run            # Run the application with uvicorn
make test           # Run tests
make test-with-coverage # Run tests with coverage report
make clean          # Clean generated files and cache
```

**Docker:**
```bash
make build          # Build Docker image
```

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
# Complete project setup
make setup

# Build Docker image
bash scripts/build-image.sh

# Build with specific tag
bash scripts/build-image.sh my-app:v1.2.3

# Build and push to registry
bash scripts/build-image.sh my-app:latest true

# Update dependencies
make upgrade-deps

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

- Installs pip-tools for dependency management
- Compiles requirements from .in files
- Installs dependencies using compiled requirements
- Sets up test environment
- Runs comprehensive test suite
- Generates coverage reports
- Validates code quality
- Tests application startup

## API Manual Testing
### With curl
#### Generate outline:
- Batch response:
   ```bash
   curl -X POST "http://localhost:8080/api/outline/generate"
      -H "Content-Type: application/json"
      -d '{
         "topic": "Introduction to Tắt đèn of Ngô Tất Tố",
         "slide_count": 5,
         "model": "gemini-2.5-flash-lite",
         "language": "vi",
         "provider": "google"
      }'
   ```
- Stream response:
   ```bash
   curl -N -X POST "http://localhost:8080/api/outline/generate/stream"
      -H "Content-Type: application/json"
      -d '{
         "topic": "Introduction to Tắt đèn of Ngô Tất Tố",
         "slide_count": 5,
         "model": "gemini-2.5-flash-lite",
         "language": "vi",
         "provider": "google",
      }'
   ```
#### Presentation generate
- Batch response:
   ```bash
   curl -X POST "http://localhost:8080/api/presentation/generate"
      -H "Content-Type: application/json"
      -d '{
            "outline": "```md\n# Introduction to Artificial Intelligence\n• What is AI?\n• Why AI matters in today&apos;s world\n• Overview of presentation goals\n\n# History and Evolution of AI\n• Early concepts and origins (1940s-1950s)\n• The AI winters and revivals\n• Major breakthroughs and milestones\n• Key pioneers and their contributions\n\n# Types and Categories of AI\n• Narrow AI vs General AI\n• Machine Learning fundamentals\n• Deep Learning and Neural Networks\n• Natural Language Processing\n• Computer Vision\n\n# Real-World Applications of AI\n• Healthcare and medical diagnosis\n• Transportation and autonomous vehicles\n• Finance and fraud detection\n• Entertainment and recommendation systems\n• Smart homes and IoT devices\n\n# Future of AI and Emerging Trends\n• Ethical considerations and responsible AI\n• AI governance and regulation\n• Potential societal impacts\n• Career opportunities in AI\n• Preparing for an AI-driven future\n```",
            "provider": "google",
            "slide_count": 5,
            "model": "gemini-2.5-flash-lite",
            "language": "vi"
        }'
   ```
- Stream response:
   ```bash
   curl -N -X POST "http://localhost:8080/api/presentation/generate/stream"
      -H "Content-Type: application/json"
      -d '{
            "outline": "```md\n# Introduction to Artificial Intelligence\n• What is AI?\n• Why AI matters in today&apos;s world\n• Overview of presentation goals\n\n# History and Evolution of AI\n• Early concepts and origins (1940s-1950s)\n• The AI winters and revivals\n• Major breakthroughs and milestones\n• Key pioneers and their contributions\n\n# Types and Categories of AI\n• Narrow AI vs General AI\n• Machine Learning fundamentals\n• Deep Learning and Neural Networks\n• Natural Language Processing\n• Computer Vision\n\n# Real-World Applications of AI\n• Healthcare and medical diagnosis\n• Transportation and autonomous vehicles\n• Finance and fraud detection\n• Entertainment and recommendation systems\n• Smart homes and IoT devices\n\n# Future of AI and Emerging Trends\n• Ethical considerations and responsible AI\n• AI governance and regulation\n• Potential societal impacts\n• Career opportunities in AI\n• Preparing for an AI-driven future\n```",
            "provider": "google",
            "slide_count": 5,
            "model": "gemini-2.5-flash-lite",
            "language": "vi"
        }'
   ```
### With web UI
- First, `cd web_test_api` folder
- Then run:
   ```bash
     python3 -m http.server 3000
   ```
- Access to `localhost:3000` in your browser
### Image Generation API

Generate images based on text descriptions. The API returns a base64-encoded image that can be displayed in a web application:

```bash
curl -X POST "http://localhost:8080/api/v1/image/generate"
    -H "Content-Type: application/json"
    -d '{
    "prompt": "A beautiful mountain landscape with a lake and trees",
    "sample_count": 1,
    "aspect_ratio": "1024x1024",
    "safety_filter_level": "BLOCK_NONE",
    "person_generation": "ALLOW",
    "seed": 42
    }'
```

The response will be just the base64-encoded image data (without any mime prefix):

```json
{
  "base64_image": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

You can use this in HTML like:

```html
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..." />
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
