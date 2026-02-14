#!/usr/bin/env python3
"""
Main ingestion script to load, process, and store documents in the vector database.

This script orchestrates the entire document ingestion pipeline:
1. Discover documents from a directory
2. For each document:
   a. Load the document
   b. Split into chunks
   c. Generate embeddings
   d. Store in PostgreSQL vector database

Usage:
    python ingestion_app/main.py --docs-dir ./data/documents --reset
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion_app.documents_chunking import DocumentChunker
from ingestion_app.documents_embedding import EmbeddingService
from ingestion_app.documents_loader import DocumentLoader
from ingestion_app.vector_store import VectorStoreManager


def load_env_config():
    """Load configuration from environment variables."""
    # Load .env file
    load_dotenv()

    config = {
        "project_id": os.getenv("VERTEX_PROJECT_ID"),
        "location": os.getenv("VERTEX_LOCATION", "us-central1"),
        "service_account_file": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-004"),
        # Hierarchical chunking configuration
        "parent_chunk_size": int(os.getenv("PARENT_CHUNK_SIZE", "2000")),
        "child_chunk_size": int(os.getenv("CHILD_CHUNK_SIZE", "500")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "100")),
        # PostgreSQL configuration
        "pg_connection_string": os.getenv("PG_CONNECTION_STRING"),
        "pg_host": os.getenv("PG_HOST", "localhost"),
        "pg_port": int(os.getenv("PG_PORT", "5432")),
        "pg_database": os.getenv("PG_DATABASE", "vectordb"),
        "pg_user": os.getenv("PG_USER", "postgres"),
        "pg_password": os.getenv("PG_PASSWORD"),
        "collection_name": os.getenv("COLLECTION_NAME", "documents"),
        # LlamaParse PDF configuration
        "pdf_language": os.getenv("PDF_LANGUAGE", "vi"),
        "use_premium_pdf_mode": os.getenv(
            "USE_PREMIUM_PDF_MODE", "true"
        ).lower()
        == "true",
    }

    # Validate required fields
    if not config["project_id"]:
        raise ValueError("VERTEX_PROJECT_ID environment variable is required")

    if not config["service_account_file"]:
        raise ValueError(
            "GOOGLE_APPLICATION_CREDENTIALS environment variable is required"
        )

    # Validate PostgreSQL configuration
    if not config["pg_connection_string"] and not config["pg_password"]:
        raise ValueError(
            "Either PG_CONNECTION_STRING or PG_PASSWORD environment variable is required"
        )

    return config


def discover_documents(
    directory_path: str, recursive: bool, supported_extensions: list
) -> list:
    """
    Discover all supported document files in a directory.

    Args:
        directory_path: Path to the directory containing documents
        recursive: Whether to search subdirectories recursively
        supported_extensions: List of supported file extensions

    Returns:
        List of file paths
    """
    dir_path = Path(directory_path)
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"Directory not found: {directory_path}")

    file_paths = []
    pattern = "**/*" if recursive else "*"

    for file_path in dir_path.glob(pattern):
        if file_path.is_file() and file_path.suffix in supported_extensions:
            file_paths.append(file_path)

    return file_paths


def main():
    """Main ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into vector database for RAG operations"
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default="./data/documents",
        help="Directory containing documents to ingest",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        help="Name of the ChromaDB collection (overrides env variable)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset/clear existing collection before ingesting",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search subdirectories recursively (default: True)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Document Ingestion Pipeline")
    print("=" * 70)

    # Load configuration
    try:
        config = load_env_config()
        if args.collection_name:
            config["collection_name"] = args.collection_name
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)

    print(f"\n📋 Configuration:")
    print(f"  Documents directory: {args.docs_dir}")
    print(f"  Collection name: {config['collection_name']}")
    print(f"  Embedding model: {config['embedding_model']}")
    print(
        f"  Parent chunk size: {config['parent_chunk_size']} chars (context)"
    )
    print(
        f"  Child chunk size: {config['child_chunk_size']} chars (embedding)"
    )
    print(f"  Chunk overlap: {config['chunk_overlap']}")
    print(f"  PostgreSQL Database: {config['pg_database']}")
    print(f"  PostgreSQL Host: {config['pg_host']}:{config['pg_port']}")
    print(f"  PDF Language: {config['pdf_language']}")
    print(f"  PDF Premium Mode: {config['use_premium_pdf_mode']}")
    print()

    # Validate database connection before proceeding
    print("🔍 Validating database connection...")
    print("-" * 70)
    try:
        import psycopg2

        # Build connection string for validation
        if config["pg_connection_string"]:
            # Convert SQLAlchemy format to psycopg2 format
            # Replace 'postgresql+psycopg2://' with 'postgresql://'
            test_conn_string = config["pg_connection_string"].replace(
                "postgresql+psycopg://", "postgresql://"
            )
        else:
            test_conn_string = (
                f"postgresql://{config['pg_user']}:{config['pg_password']}@"
                f"{config['pg_host']}:{config['pg_port']}/{config['pg_database']}"
            )

        # Attempt to connect
        conn = psycopg2.connect(test_conn_string)
        cursor = conn.cursor()

        # Check PostgreSQL version
        cursor.execute("SELECT version();")
        pg_version = cursor.fetchone()[0]
        print(f"✓ Connected to PostgreSQL")
        print(f"  Version: {pg_version.split(',')[0]}")

        # Check if the document_embeddings table exists
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = %s
            );
            """,
            (config["collection_name"],),
        )
        table_exists = cursor.fetchone()[0]

        if table_exists:
            # Count existing documents
            cursor.execute(
                f"SELECT COUNT(*) FROM {config['collection_name']};"
            )
            doc_count = cursor.fetchone()[0]
            print(
                f"✓ Collection '{config['collection_name']}' exists ({doc_count} documents)"
            )
        else:
            print(
                f"ℹ️  Collection '{config['collection_name']}' does not exist yet (will be created)"
            )

        # Check if pgvector extension exists
        cursor.execute(
            "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');"
        )
        has_pgvector = cursor.fetchone()[0]

        if has_pgvector:
            print(f"✓ pgvector extension is installed")
        else:
            print(f"⚠️  pgvector extension is NOT installed")
            print(
                f"   The pipeline will attempt to create it (requires superuser privileges)"
            )
            print(f"   If creation fails, please install pgvector manually:")
            print(
                f"   - Ubuntu/Debian: sudo apt install postgresql-17-pgvector"
            )
            print(
                f"   - Or follow: https://github.com/pgvector/pgvector#installation"
            )

        cursor.close()
        conn.close()
        print()

    except psycopg2.OperationalError as e:
        print(f"❌ Failed to connect to PostgreSQL database")
        print(f"   Error: {e}")
        print(f"\n   Please check:")
        print(
            f"   1. PostgreSQL is running on {config['pg_host']}:{config['pg_port']}"
        )
        print(f"   2. Database '{config['pg_database']}' exists")
        print(f"   3. User '{config['pg_user']}' has correct password")
        print(f"   4. PostgreSQL accepts connections from this host")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Database validation error: {e}")
        sys.exit(1)

    # Initialize services
    print("🔧 Initializing services...")
    print("-" * 70)

    # Initialize document loader
    try:
        loader = DocumentLoader(
            pdf_language=config["pdf_language"],
            use_premium_mode=config["use_premium_pdf_mode"],
        )
        print(f"✓ Document loader initialized")
    except Exception as e:
        print(f"❌ Error initializing document loader: {e}")
        sys.exit(1)

    # Initialize chunker
    try:
        chunker = DocumentChunker(
            parent_chunk_size=config["parent_chunk_size"],
            child_chunk_size=config["child_chunk_size"],
            chunk_overlap=config["chunk_overlap"],
        )
        print(f"✓ Document chunker initialized")
    except Exception as e:
        print(f"❌ Error initializing chunker: {e}")
        sys.exit(1)

    # Initialize embedding service
    try:
        embedding_service = EmbeddingService(
            model_name=config["embedding_model"],
            project_id=config["project_id"],
            location=config["location"],
            service_account_file=config["service_account_file"],
        )
        print(f"✓ Embedding service initialized ({config['embedding_model']})")
        print(
            f"  Embedding dimension: {embedding_service.get_embedding_dimension()}"
        )
    except Exception as e:
        print(f"❌ Error initializing embeddings: {e}")
        sys.exit(1)

    # Initialize vector store
    try:
        if config["pg_connection_string"]:
            connection_string = config["pg_connection_string"].replace(
                "postgresql+psycopg://", "postgresql+psycopg2://"
            )
        else:
            connection_string = None

        vector_store = VectorStoreManager(
            embeddings=embedding_service.get_embeddings(),
            collection_name=config["collection_name"],
            connection_string=connection_string,
            host=config["pg_host"],
            port=config["pg_port"],
            database=config["pg_database"],
            user=config["pg_user"],
            password=config["pg_password"],
        )

        # Try to create pgvector extension
        if connection_string:
            VectorStoreManager.create_extension(connection_string)
        else:
            conn_str = f"postgresql+psycopg2://{config['pg_user']}:{config['pg_password']}@{config['pg_host']}:{config['pg_port']}/{config['pg_database']}"
            VectorStoreManager.create_extension(conn_str)

        if args.reset:
            vector_store.reset_collection()

        print(f"✓ PostgreSQL vector store ready")
    except Exception as e:
        print(f"❌ Error initializing vector store: {e}")
        sys.exit(1)

    # Step 1: Discover documents
    print("\n📚 Step 1: Discovering documents...")
    print("-" * 70)
    try:
        file_paths = discover_documents(
            args.docs_dir, args.recursive, loader.get_supported_extensions()
        )

        if not file_paths:
            print(f"⚠️  No documents found in {args.docs_dir}")
            print(
                f"Supported formats: {', '.join(loader.get_supported_extensions())}"
            )
            sys.exit(0)

        print(f"✓ Found {len(file_paths)} document(s) to process")
        for fp in file_paths:
            print(f"  - {fp.name}")
    except Exception as e:
        print(f"❌ Error discovering documents: {e}")
        sys.exit(1)

    # Step 2: Process each document individually
    print("\n🔄 Step 2: Processing documents individually...")
    print("-" * 70)

    total_chunks_stored = 0
    successful_docs = 0
    failed_docs = 0

    for idx, file_path in enumerate(file_paths, 1):
        print(f"\n[{idx}/{len(file_paths)}] Processing: {file_path.name}")
        print("  " + "-" * 66)

        try:
            # 2a. Load document
            print(f"  📄 Loading document...")
            documents = loader.load_file(str(file_path))
            if not documents:
                print(f"  ⚠️  No content loaded, skipping")
                failed_docs += 1
                continue
            print(f"  ✓ Loaded {len(documents)} page(s)")

            # Display educational metadata if available
            if (
                documents
                and documents[0].metadata.get("has_metadata") is not False
            ):
                metadata_summary = documents[0].metadata.get(
                    "metadata_summary"
                )
                if metadata_summary:
                    print(f"  📚 {metadata_summary}")

            # 2b. Chunk document
            print(f"  ✂️  Chunking document...")
            parent_chunks, child_chunks = chunker.split_documents(documents)
            stats = chunker.get_chunk_stats(parent_chunks, child_chunks)
            print(f"  ✓ Created {stats['total_parent_chunks']} parent chunks")
            print(
                f"  ✓ Created {stats['total_child_chunks']} child chunks (for embedding)"
            )

            # Use child chunks for embedding (they contain parent context in metadata)
            chunks = child_chunks

            # 2c. Embed and store chunks
            print(f"  💾 Storing chunks in vector database...")
            doc_ids = vector_store.add_documents(chunks, batch_size=100)
            print(f"  ✓ Stored {len(doc_ids)} chunks")

            total_chunks_stored += len(doc_ids)
            successful_docs += 1

        except Exception as e:
            print(f"  ❌ Error processing document: {e}")
            failed_docs += 1
            continue

    # Success summary
    print("\n" + "=" * 70)
    if failed_docs == 0:
        print("✅ Ingestion completed successfully!")
    else:
        print("⚠️  Ingestion completed with some errors")
    print("=" * 70)
    print(f"\n📝 Summary:")
    print(f"  Total documents found: {len(file_paths)}")
    print(f"  Successfully processed: {successful_docs}")
    print(f"  Failed: {failed_docs}")
    print(f"  Total chunks stored: {total_chunks_stored}")
    print(f"  Collection: {config['collection_name']}")
    print()


if __name__ == "__main__":
    main()
