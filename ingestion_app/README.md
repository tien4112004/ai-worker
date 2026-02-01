# Document Ingestion Module

This module handles the ingestion of documents into a PostgreSQL vector database with pgvector for RAG (Retrieval Augmented Generation) operations.

## Features

- **Document Loading**: Support for PDF, TXT, DOCX, and Markdown files
- **Document Cleaning**: Preprocessing with configurable options
- **Text Chunking**: Intelligent splitting with configurable size and overlap
- **Embeddings**: Google Vertex AI embeddings (text-embedding-004)
- **Vector Storage**: PostgreSQL with pgvector extension for efficient similarity search

## Components

### 1. DocumentLoader (`documents_loader.py`)

Loads documents from various file formats with advanced PDF parsing capabilities.

**Supported formats:**

- PDF (`.pdf`) - Uses **LlamaParse** for advanced parsing with OCR and table recognition
- Text (`.txt`)
- Markdown (`.md`)
- Word Documents (`.docx`)

**PDF Features (LlamaParse):**
- Advanced OCR for scanned documents
- Table recognition and markdown conversion
- Multi-language support (Vietnamese by default)
- Layout analysis and structure preservation
- Configurable parsing instructions
- Automatic fallback to basic loader if LlamaParse unavailable

### 2. DocumentCleaner (`documents_cleaning.py`)

Preprocesses documents to improve quality:

- Remove extra whitespace
- Remove URLs and emails (optional)
- Normalize text (optional)
- Remove duplicates

### 3. DocumentChunker (`documents_chunking.py`)

Splits documents into chunks for optimal embedding:

- Configurable chunk size (default: 1000 characters)
- Configurable overlap (default: 200 characters)
- Hierarchical splitting (paragraphs → sentences → words → characters)

### 4. EmbeddingService (`documents_embedding.py`)

Generates embeddings using Google Vertex AI:

- Model: text-embedding-004 (768 dimensions)
- Batch and single query embedding
- Async support

### 5. VectorStoreManager (`vector_store.py`)

Manages PostgreSQL with pgvector extension:

- Document addition with batching
- Similarity search with scores
- JSONB metadata filtering
- Retriever creation for RAG operations
- Collection management (reset, delete, stats)

## Installation

### 1. Set up PostgreSQL with pgvector

See [PGVECTOR_SETUP.md](PGVECTOR_SETUP.md) for detailed setup instructions.

**Quick start with Docker:**

```bash
docker run -d \
  --name pgvector-db \
  -e POSTGRES_PASSWORD=yourpassword \
  -e POSTGRES_DB=vectordb \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### 2. Install Python dependencies

```bash
# From project root
cd /home/ltt204/graduation-projec/ai-worker

# Compile and install dependencies
make compile-deps
make sync-deps
```

### 3. Configure environment variables

Edit your `.env` file:

```bash
# Vertex AI Configuration
VERTEX_PROJECT_ID=your-project-id
VERTEX_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=./service-account.json

# PostgreSQL Configuration
PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=vectordb
PG_USER=postgres
PG_PASSWORD=yourpassword

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-004

# Chunking Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Collection Configuration
COLLECTION_NAME=documents

# LlamaParse Configuration (for advanced PDF parsing)
LLAMA_CLOUD_API_KEY=your-llama-cloud-api-key
PDF_LANGUAGE=vi
USE_PREMIUM_PDF_MODE=true
```

**LlamaParse Configuration:**
- `LLAMA_CLOUD_API_KEY`: Get your API key from [LlamaIndex Cloud](https://cloud.llamaindex.ai/)
- `PDF_LANGUAGE`: Language code for OCR (e.g., "vi" for Vietnamese, "en" for English)
- `USE_PREMIUM_PDF_MODE`: Enable premium mode for better accuracy (true/false)

**Note:** If LlamaParse is not configured or fails, the loader automatically falls back to the basic PyMuPDF loader.

## Usage

### Basic Usage

```bash
# Ingest documents from default directory (./data/documents)
python ingestion_app/main.py

# Specify custom directory
python ingestion_app/main.py --docs-dir /path/to/your/documents

# Reset collection before ingesting
python ingestion_app/main.py --reset

# Use custom collection name
python ingestion_app/main.py --collection-name my_docs
```

### Programmatic Usage

```python
from ingestion_app.documents_loader import DocumentLoader
from ingestion_app.documents_chunking import DocumentChunker
from ingestion_app.documents_embedding import EmbeddingService
from ingestion_app.vector_store import VectorStoreManager

# 1. Load documents with LlamaParse for PDFs
loader = DocumentLoader(
    pdf_language="vi",  # Vietnamese for PDFs
    use_premium_mode=True,  # Use LlamaParse premium mode
    parsing_instruction="Custom parsing instructions..."  # Optional
)
documents = loader.load_from_directory("./data/documents")

# 2. Chunk documents
chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
chunks = chunker.split_documents(documents)

# 3. Initialize embedding service
embedding_service = EmbeddingService(
    model_name="text-embedding-004",
    project_id="your-project-id",
    location="us-central1",
    service_account_file="./service-account.json"
)

# 4. Create vector store
vector_store = VectorStoreManager(
    embeddings=embedding_service.get_embeddings(),
    collection_name="documents",
    host="localhost",
    port=5432,
    database="vectordb",
    user="postgres",
    password="yourpassword"
)

# 5. Add documents
doc_ids = vector_store.add_documents(chunks)

# 6. Search for similar documents
results = vector_store.similarity_search("your query", k=4)
```

## Directory Structure

```
ingestion_app/
├── __init__.py              # Package initialization
├── main.py                  # Main ingestion script
├── documents_loader.py      # Document loading
├── documents_cleaning.py    # Document preprocessing
├── documents_chunking.py    # Text chunking
├── documents_embedding.py   # Embedding generation
├── vector_store.py          # PostgreSQL/PGVector management
├── README.md               # This file
└── PGVECTOR_SETUP.md       # PGVector setup guide

data/
├── documents/              # Source documents (you create this)
│   ├── sample.pdf
│   ├── guide.docx
│   └── readme.txt
```

## Querying the Database

You can query your vector database directly using SQL:

```sql
-- View collections
SELECT * FROM langchain_pg_collection;

-- Count documents
SELECT COUNT(*) FROM langchain_pg_embedding;

-- Search by metadata
SELECT document, cmetadata
FROM langchain_pg_embedding
WHERE cmetadata->>'source_file' = 'ai_overview.txt';
```

## Testing

See sample documents in `data/documents/` for testing the ingestion pipeline.

## Migration from ChromaDB

If you previously used ChromaDB:

1. **Update dependencies**: Done automatically via requirements.in
2. **Configure PostgreSQL**: Set up PGVector (see PGVECTOR_SETUP.md)
3. **Update .env**: Add PostgreSQL connection details
4. **Run ingestion**: Use `--reset` flag to start fresh

The vector_store.py API remains compatible, so no changes needed in RAG service code.

## Next Steps

After running ingestion, you can:

1. Use the vector store for RAG operations
2. Integrate with the RAG service in the main app
3. Query documents via the API endpoints

## Troubleshooting

### "Extension 'vector' does not exist"

- Make sure pgvector is installed in your PostgreSQL instance
- Run `CREATE EXTENSION vector;` as a superuser

### Connection errors

- Check PostgreSQL is running: `sudo systemctl status postgresql`
- Verify connection details in `.env`
- Test connection: `psql -h localhost -U postgres -d vectordb`

### Memory issues

- Reduce batch size in ingestion script
- Increase PostgreSQL shared_buffers in postgresql.conf
