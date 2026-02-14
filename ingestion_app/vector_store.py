"""Vector store module to manage PostgreSQL with pgvector for document storage and retrieval."""

from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorStoreManager:
    """Manages PostgreSQL with pgvector extension for document retrieval."""

    def __init__(
        self,
        embeddings: Embeddings,
        collection_name: str = "documents",
        connection_string: Optional[str] = None,
        host: Optional[str] = "localhost",
        port: int = 5432,
        database: Optional[str] = "vectordb",
        user: Optional[str] = "postgres",
        password: Optional[str] = None,
    ):
        """
        Initialize the VectorStoreManager with PostgreSQL + pgvector.

        Args:
            embeddings: LangChain embeddings instance
            collection_name: Name of the table/collection in PostgreSQL
            connection_string: Full PostgreSQL connection string (overrides other params)
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            user: Database user
            password: Database password
        """
        self.embeddings = embeddings
        self.collection_name = collection_name

        if connection_string:
            self.connection_string = connection_string
        else:
            self.connection_string = (
                f"postgresql://{user}:{password}@{host}:{port}/{database}"
            )

        # Initialize the vector store
        self.vector_store = self._initialize_store()

    def _initialize_store(self) -> PGVector:
        """
        Initialize or connect to PostgreSQL with pgvector.

        Returns:
            PGVector vector store instance
        """
        return PGVector(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            connection_string=self.connection_string,
            use_jsonb=True,  # Store metadata as JSONB for better querying
        )

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects to add
            batch_size: Number of documents to process in each batch
            ids: Optional list of IDs for the documents

        Returns:
            List of document IDs
        """
        if not documents:
            return []

        print(
            f"Adding {len(documents)} documents to PostgreSQL vector store..."
        )

        # Process in batches to avoid memory issues
        all_ids = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_ids = ids[i : i + batch_size] if ids else None

            doc_ids = self.vector_store.add_documents(
                documents=batch,
                ids=batch_ids,
            )
            all_ids.extend(doc_ids)

            print(
                f"  Processed batch {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1}"
            )

        print(f"✓ Successfully added {len(all_ids)} documents to PostgreSQL")
        return all_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform similarity search for relevant documents.

        Args:
            query: Query text
            k: Number of documents to return
            filter: Optional metadata filter (JSONB query)

        Returns:
            List of relevant documents
        """
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.

        Args:
            query: Query text
            k: Number of documents to return
            filter: Optional metadata filter

        Returns:
            List of (document, score) tuples
        """
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
        )

    def get_retriever(
        self,
        k: int = 4,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Get a retriever for RAG operations.

        Args:
            k: Number of documents to retrieve
            search_type: Type of search ("similarity", "mmr", or "similarity_score_threshold")
            search_kwargs: Additional search parameters

        Returns:
            VectorStoreRetriever instance
        """
        if search_kwargs is None:
            search_kwargs = {"k": k}
        else:
            search_kwargs["k"] = k

        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

    def delete_collection(self):
        """Delete the entire collection/table."""
        try:
            self.vector_store.delete_collection()
            print(f"✓ Deleted collection '{self.collection_name}'")
        except Exception as e:
            print(f"⚠️  Error deleting collection: {e}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            # Execute a query to count documents
            import psycopg2

            psycopg2_url = self.connection_string.replace(
                "postgresql+psycopg2://", "postgresql://"
            )
            conn = psycopg2.connect(psycopg2_url)
            cursor = conn.cursor()

            # Count total documents in the table
            cursor.execute(
                f"SELECT COUNT(*) FROM langchain_pg_embedding WHERE collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = %s)",
                (self.collection_name,),
            )
            count = cursor.fetchone()[0]

            cursor.close()
            conn.close()

            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "database": self.connection_string.split("/")[-1].split("?")[
                    0
                ],
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "error": str(e),
                "database": self.connection_string.split("/")[-1].split("?")[
                    0
                ],
            }

    def reset_collection(self):
        """Delete and reinitialize the collection."""
        print(f"Resetting collection '{self.collection_name}'...")
        self.delete_collection()
        self.vector_store = self._initialize_store()
        print("✓ Collection reset complete")

    @staticmethod
    def create_extension(connection_string: str):
        """
        Create the pgvector extension in PostgreSQL if it doesn't exist.

        Args:
            connection_string: PostgreSQL connection string

        Note: This requires superuser privileges
        """
        import psycopg2

        try:
            psycopg2_url = connection_string.replace(
                "postgresql+psycopg2://", "postgresql://"
            )
            conn = psycopg2.connect(psycopg2_url)
            conn.autocommit = True
            cursor = conn.cursor()

            # Create extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            print("✓ pgvector extension created/verified")

            cursor.close()
            conn.close()
        except Exception as e:
            print(f"⚠️  Error creating pgvector extension: {e}")
            print(
                "  Make sure you have superuser privileges or the extension is already installed"
            )
