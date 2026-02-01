"""Embedding service module to generate embeddings using Vertex AI."""

import json
from typing import List, Optional

import vertexai
from google.oauth2 import service_account
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class EmbeddingService:
    """Manages document embeddings using Google Vertex AI."""

    def __init__(
        self,
        model_name: str = "gemini-embedding-001",
        project_id: Optional[str] = None,
        location: Optional[str] = "us-central1",
        service_account_file: Optional[str] = None,
    ):
        """
        Initialize the EmbeddingService.

        Args:
            model_name: Name of the Vertex AI embedding model
            project_id: Google Cloud project ID
            location: Google Cloud location/region
            service_account_file: Path to service account JSON file
        """
        self.model_name = model_name
        self.project_id = project_id
        self.location = location
        self.service_account_file = service_account_file

        # Initialize Vertex AI
        self._initialize_vertexai()

        # Create embeddings instance
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            project=project_id,
            location=location,
        )

    def _initialize_vertexai(self):
        """Initialize Vertex AI with credentials."""
        if self.service_account_file:
            # Load credentials from service account file
            service_account_info = json.load(open(self.service_account_file))
            credentials = (
                service_account.Credentials.from_service_account_info(
                    service_account_info
                )
            )

            # Initialize Vertex AI with credentials
            vertexai.init(
                project=self.project_id,
                location=self.location,
                credentials=credentials,
            )
        else:
            # Use default credentials
            vertexai.init(
                project=self.project_id,
                location=self.location,
            )

    def get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """
        Get the LangChain embeddings instance.

        Returns:
            VertexAIEmbeddings instance
        """
        return self.embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.

        Args:
            texts: List of text documents to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously generate embeddings for documents.

        Args:
            texts: List of text documents to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        return await self.embeddings.aembed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """
        Asynchronously generate embedding for a query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        return await self.embeddings.aembed_query(text)

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            Embedding dimension (768 for text-embedding-004)
        """
        # text-embedding-004 produces 768-dimensional embeddings
        if "text-embedding-004" in self.model_name:
            return 768
        # Default dimension for most Vertex AI embedding models
        return 768
