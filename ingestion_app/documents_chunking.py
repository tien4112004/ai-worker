"""Hierarchical chunking module to split documents into parent-child chunks for embedding."""

from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class HierarchicalDocumentChunker:
    """Splits documents into hierarchical parent-child chunks for optimal retrieval."""

    def __init__(
        self,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
        length_function: callable = len,
    ):
        """
        Initialize the Hierarchical Document Chunker.

        Args:
            parent_chunk_size: Maximum size of parent chunks (larger context)
            child_chunk_size: Maximum size of child chunks (for embedding)
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separator strings to split on (default: hierarchical split)
            length_function: Function to measure chunk length (default: len)
        """
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.chunk_overlap = chunk_overlap

        # Use default hierarchical separators if not provided
        if separators is None:
            separators = [
                "\n\n",  # Paragraph breaks
                "\n",  # Line breaks
                ". ",  # Sentence breaks
                " ",  # Word breaks
                "",  # Character-level fallback
            ]

        # Parent splitter - creates larger context chunks
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=length_function,
            is_separator_regex=False,
        )

        # Child splitter - creates smaller chunks for embedding
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=length_function,
            is_separator_regex=False,
        )

    def split_documents(
        self, documents: List[Document]
    ) -> Tuple[List[Document], List[Document]]:
        """
        Split documents into hierarchical parent-child chunks.

        Args:
            documents: List of Document objects to split

        Returns:
            Tuple of (parent_chunks, child_chunks) where child chunks reference their parents
        """
        if not documents:
            return [], []

        all_parent_chunks = []
        all_child_chunks = []
        global_parent_id = 0
        global_child_id = 0

        for doc in documents:
            # Create parent chunks
            parent_chunks = self.parent_splitter.split_documents([doc])

            for parent_idx, parent_chunk in enumerate(parent_chunks):
                # Add parent metadata
                parent_chunk.metadata["chunk_id"] = global_parent_id
                parent_chunk.metadata["chunk_type"] = "parent"
                parent_chunk.metadata["chunk_size"] = len(
                    parent_chunk.page_content
                )
                parent_chunk.metadata["parent_id"] = global_parent_id
                parent_chunk.metadata["doc_id"] = doc.metadata.get(
                    "doc_id", "unknown"
                )

                # Create child chunks from this parent
                child_chunks = self.child_splitter.split_text(
                    parent_chunk.page_content
                )

                for child_idx, child_text in enumerate(child_chunks):
                    # Create child document with reference to parent
                    child_metadata = {
                        **parent_chunk.metadata,
                        "chunk_id": global_child_id,
                        "chunk_type": "child",
                        "chunk_size": len(child_text),
                        "parent_id": global_parent_id,
                        "parent_text": parent_chunk.page_content,  # Store parent context
                        "child_index": child_idx,
                    }

                    child_doc = Document(
                        page_content=child_text, metadata=child_metadata
                    )
                    all_child_chunks.append(child_doc)
                    global_child_id += 1

                all_parent_chunks.append(parent_chunk)
                global_parent_id += 1

        return all_parent_chunks, all_child_chunks

    def split_text(
        self, text: str, metadata: Optional[dict] = None
    ) -> Tuple[List[Document], List[Document]]:
        """
        Split raw text into hierarchical parent-child chunks.

        Args:
            text: Raw text to split
            metadata: Optional metadata to attach to all chunks

        Returns:
            Tuple of (parent_chunks, child_chunks)
        """
        metadata = metadata or {}
        doc = Document(page_content=text, metadata=metadata)
        return self.split_documents([doc])

    def get_chunk_stats(
        self, parent_chunks: List[Document], child_chunks: List[Document]
    ) -> dict:
        """
        Get statistics about hierarchical chunks.

        Args:
            parent_chunks: List of parent chunks
            child_chunks: List of child chunks

        Returns:
            Dictionary with chunk statistics
        """
        if not parent_chunks and not child_chunks:
            return {
                "total_parent_chunks": 0,
                "total_child_chunks": 0,
                "avg_parent_size": 0,
                "avg_child_size": 0,
                "avg_children_per_parent": 0,
                "total_characters": 0,
            }

        parent_sizes = [len(doc.page_content) for doc in parent_chunks]
        child_sizes = [len(doc.page_content) for doc in child_chunks]

        return {
            "total_parent_chunks": len(parent_chunks),
            "total_child_chunks": len(child_chunks),
            "avg_parent_size": (
                sum(parent_sizes) // len(parent_sizes) if parent_sizes else 0
            ),
            "avg_child_size": (
                sum(child_sizes) // len(child_sizes) if child_sizes else 0
            ),
            "min_parent_size": min(parent_sizes) if parent_sizes else 0,
            "max_parent_size": max(parent_sizes) if parent_sizes else 0,
            "min_child_size": min(child_sizes) if child_sizes else 0,
            "max_child_size": max(child_sizes) if child_sizes else 0,
            "avg_children_per_parent": (
                len(child_chunks) / len(parent_chunks) if parent_chunks else 0
            ),
            "total_characters": sum(parent_sizes),
        }


# Backward compatibility alias
DocumentChunker = HierarchicalDocumentChunker
