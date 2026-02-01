# PostgreSQL with PGVector Setup Guide

This guide explains how to set up PostgreSQL with the pgvector extension for the document ingestion module.

## What is PGVector?

PGVector is a PostgreSQL extension that adds vector similarity search capabilities to PostgreSQL. It allows you to store and search high-dimensional vectors efficiently, making it perfect for RAG applications.

### Benefits over ChromaDB

- **Production-ready**: PostgreSQL is battle-tested and widely used in production
- **ACID compliance**: Full transactional support
- **JSON metadata**: Query documents using JSONB fields
- **Scalability**: Horizontal scaling with replicas
- **Familiar tooling**: Standard SQL, pgAdmin, psql, etc.

## Installation

### Option 1: Docker (Recommended for Development)

The easiest way to get started is using Docker with the official pgvector image:

```bash
# Pull the pgvector-enabled PostgreSQL image
docker pull pgvector/pgvector:pg16

# Run PostgreSQL with pgvector
docker run -d \
  --name pgvector-db \
  -e POSTGRES_PASSWORD=yourpassword \
  -e POSTGRES_DB=vectordb \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### Option 2: Install on Existing PostgreSQL

If you have PostgreSQL already installed:

#### Ubuntu/Debian

```bash
sudo apt-get install postgresql-16-pgvector
```

#### macOS (Homebrew)

```bash
brew install pgvector
```

#### From Source

```bash
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
make install  # May require sudo
```

After installation, enable the extension in your database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

## Configuration

### 1. Create Database

Connect to PostgreSQL and create a database for vectors:

```sql
CREATE DATABASE vectordb;
```

### 2. Enable pgvector Extension

```sql
\c vectordb
CREATE EXTENSION IF NOT EXISTS vector;
```

### 3. Configure Environment Variables

Update your `.env` file with PostgreSQL connection details:

```bash
# PostgreSQL Configuration
PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=vectordb
PG_USER=postgres
PG_PASSWORD=yourpassword

# Or use connection string directly
PG_CONNECTION_STRING=postgresql://postgres:yourpassword@localhost:5432/vectordb

# Vector Store Configuration
COLLECTION_NAME=documents
EMBEDDING_MODEL=text-embedding-004
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## Usage

### Running Ingestion

After setting up PostgreSQL and configuring environment variables:

```bash
# Run ingestion (the script will create tables automatically)
python ingestion_app/main.py --docs-dir ./data/documents --reset
```

The `--reset` flag will drop and recreate the collection table.

### Database Schema

The ingestion module uses LangChain's PGVector integration, which creates these tables:

- `langchain_pg_collection`: Stores collection metadata
- `langchain_pg_embedding`: Stores document embeddings and content

Example query to view your documents:

```sql
-- View all collections
SELECT * FROM langchain_pg_collection;

-- Count documents in a collection
SELECT COUNT(*)
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON e.collection_id = c.uuid
WHERE c.name = 'documents';

-- Search for documents by metadata
SELECT
  e.document,
  e.cmetadata,
  e.embedding
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON e.collection_id = c.uuid
WHERE c.name = 'documents'
  AND e.cmetadata->>'source_file' = 'ai_overview.txt'
LIMIT 10;
```

## Production Deployment

### Docker Compose

Create a `docker-compose.pgvector.yml`:

```yaml
version: '3.8'

services:
  pgvector:
    image: pgvector/pgvector:pg16
    container_name: pgvector-db
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${PG_PASSWORD}
      POSTGRES_DB: vectordb
    ports:
      - '5432:5432'
    volumes:
      - pgvector_data:/var/lib/postgresql/data
    healthcheck:
      test: ['CMD-SHELL', 'pg_isready -U postgres']
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  pgvector_data:
```

Run with:

```bash
docker-compose -f docker-compose.pgvector.yml up -d
```

### Performance Tuning

For better performance with large datasets:

```sql
-- Create index on embedding vectors
CREATE INDEX ON langchain_pg_embedding
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Or use HNSW index (faster but uses more memory)
CREATE INDEX ON langchain_pg_embedding
USING hnsw (embedding vector_cosine_ops);
```

### Connection Pooling

For production, consider using connection pooling with PgBouncer:

```bash
# Install PgBouncer
sudo apt-get install pgbouncer

# Update connection string to use PgBouncer
PG_CONNECTION_STRING=postgresql://postgres:password@localhost:6432/vectordb
```

## Troubleshooting

### Extension Not Found

If you get "extension does not exist" error:

```sql
-- Check available extensions
SELECT * FROM pg_available_extensions WHERE name = 'vector';

-- If missing, install pgvector package for your PostgreSQL version
```

### Connection Refused

Check PostgreSQL is running:

```bash
# Check status
sudo systemctl status postgresql

# Start if needed
sudo systemctl start postgresql

# Or for Docker
docker ps | grep pgvector
```

### Permission Denied

Make sure your user has superuser privileges to create extensions:

```sql
ALTER USER postgres WITH SUPERUSER;
```

## Backup and Restore

### Backup

```bash
pg_dump -U postgres -d vectordb > vectordb_backup.sql
```

### Restore

```bash
psql -U postgres -d vectordb < vectordb_backup.sql
```

## Next Steps

1. Set up PostgreSQL with pgvector
2. Configure your `.env` file
3. Run the ingestion script
4. Proceed with RAG service integration

For more information, visit:

- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [LangChain PGVector Documentation](https://python.langchain.com/docs/integrations/vectorstores/pgvector)
