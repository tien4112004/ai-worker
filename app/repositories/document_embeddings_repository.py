"""Repository for managing document embeddings and vector store operations."""

import json
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import vertexai
from google.oauth2 import service_account
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class DocumentEmbeddingsRepository:
    """
    Repository for managing document embeddings and vector store operations.

    This repository abstracts database operations for document storage and retrieval
    using PostgreSQL with pgvector extension and Vertex AI embeddings.
    """

    def __init__(
        self,
        pg_connection_string: str,
        embedding_model: str = "text-embedding-004",
        collection_name: str = "langchain_pg_embedding",
        vertex_project_id: Optional[str] = None,
        vertex_location: str = "us-central1",
        service_account_file: Optional[str] = None,
    ):
        """
        Initialize the DocumentEmbeddingsRepository.

        Args:
            embedding_model: Vertex AI embedding model name
            collection_name: Vector store collection name
            pg_connection_string: Full PostgreSQL connection string
            vertex_project_id: Google Cloud project ID
            vertex_location: Google Cloud location
            service_account_file: Path to service account JSON
            pg_password: PostgreSQL password
            pg_connection_string: Full connection string (overrides other pg params)
            vertex_project_id: Google Cloud project ID
            vertex_location: Google Cloud location
            service_account_file: Path to service account JSON
        """
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.vertex_project_id = vertex_project_id
        self.vertex_location = vertex_location
        self.service_account_file = service_account_file
        self.connection_string = pg_connection_string

        # Initialize components
        self._embeddings: Optional[Embeddings] = None
        self._vector_store: Optional[PGVector] = None

    def _get_embeddings(self) -> Embeddings:
        """
        Get or initialize the embeddings instance.

        Returns:
            VertexAIEmbeddings instance
        """
        if self._embeddings is None:
            # Initialize Vertex AI
            if self.service_account_file:
                service_account_info = json.load(
                    open(self.service_account_file)
                )
                credentials = (
                    service_account.Credentials.from_service_account_info(
                        service_account_info
                    )
                )
                vertexai.init(
                    project=self.vertex_project_id,
                    location=self.vertex_location,
                    credentials=credentials,
                )
            else:
                vertexai.init(
                    project=self.vertex_project_id,
                    location=self.vertex_location,
                )

            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=self.embedding_model,
                project=self.vertex_project_id,
                location=self.vertex_location,
            )

        return self._embeddings

    def _get_vector_store(self) -> PGVector:
        """
        Get or initialize the vector store instance.

        Returns:
            PGVector vector store instance
        """
        if self._vector_store is None:
            embeddings = self._get_embeddings()
            self._vector_store = PGVector(
                embedding_function=embeddings,
                collection_name=self.collection_name,
                connection_string=self.connection_string,
                use_jsonb=True,
            )
        return self._vector_store

    def similarity_search(
        self,
        query: str,
        k: int = 31,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform similarity search for relevant documents.

        Args:
            query: Query text
            k: Number of documents to return
            filter: Optional metadata filter (applied in Python to avoid
                    jsonb_path_match type-casting issues in langchain_community)

        Returns:
            List of relevant documents
        """
        vector_store = self._get_vector_store()

        # Fetch extra candidates when filtering, so we have enough after filtering
        fetch_k = k * 3 if filter else k
        docs = vector_store.similarity_search(query=query, k=fetch_k)

        if filter:
            docs = [
                doc
                for doc in docs
                if all(
                    doc.metadata.get(key) == value
                    for key, value in filter.items()
                )
            ]
            docs = docs[:k]

        return docs

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.

        Args:
            query: Query text
            k: Number of documents to return
            filter: Optional metadata filter (applied in Python)

        Returns:
            List of (document, score) tuples
        """
        vector_store = self._get_vector_store()

        fetch_k = k * 3 if filter else k
        results = vector_store.similarity_search_with_score(
            query=query, k=fetch_k
        )

        if filter:
            results = [
                (doc, score)
                for doc, score in results
                if all(
                    doc.metadata.get(key) == value
                    for key, value in filter.items()
                )
            ]
            results = results[:k]

        return results

    def get_retriever(
        self,
        k: int = 4,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
        grade: Optional[int] = None,
        subject_code: Optional[str] = None,
    ) -> BaseRetriever:
        """
        Get a retriever for RAG operations.

        Args:
            k: Number of documents to retrieve
            search_type: Type of search ("similarity", "mmr", or "similarity_score_threshold")
            search_kwargs: Additional search parameters
            grade: Optional grade level filter (1-5)
            subject_code: Optional subject code filter (T, TA, or TV)

        Returns:
            BaseRetriever instance
        """
        vector_store = self._get_vector_store()

        if search_kwargs is None:
            search_kwargs = {"k": k}
        else:
            search_kwargs = search_kwargs.copy()
            search_kwargs["k"] = k

        return vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        embeddings = self._get_embeddings()
        return embeddings.embed_query(text)

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            psycopg2_url = self.connection_string.replace(
                "postgresql+psycopg2://", "postgresql://"
            )
            conn = psycopg2.connect(psycopg2_url)
            cursor = conn.cursor()

            # Count total documents in the collection
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM langchain_pg_embedding
                WHERE collection_id = (
                    SELECT uuid
                    FROM langchain_pg_collection
                    WHERE name = %s
                )
                """,
                (self.collection_name,),
            )

            result = cursor.fetchone()
            count = result[0] if result else 0

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

    def delete_collection(self) -> None:
        """Delete the entire collection/table."""
        vector_store = self._get_vector_store()
        vector_store.delete_collection()
