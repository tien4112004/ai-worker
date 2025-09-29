FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY .env ./app/.env
COPY service-account.json ./service-account.json

# Create non-root user BEFORE changing ownership
RUN adduser --disabled-password --gecos '' appuser \
    && chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=5s --timeout=10s --start-period=2s --retries=3 \
    CMD curl -f http://localhost:8081/docs || exit 1

# Expose port
EXPOSE 8081

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8081"]
