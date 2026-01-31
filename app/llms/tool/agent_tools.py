from typing import Any, Dict, Optional

from langchain.tools import tool

from app.repositories.document_embeddings_repository import (
    DocumentEmbeddingsRepository,
)

_repository: Optional[DocumentEmbeddingsRepository] = None
_filters: Optional[Dict[str, Any]] = None


def set_global_repository(repository: DocumentEmbeddingsRepository):
    """Set the global repository instance."""
    global _repository
    _repository = repository


def set_search_filters(filters: Optional[Dict[str, Any]] = None):
    """
    Set filters for document search.

    Args:
        filters: Dictionary with filter criteria (e.g., {"subject_code": "T", "grade": "5"})
                 subject_code can be: 'T' (Toán), 'TV' (Tiếng Việt), 'TA' (Tiếng Anh)
    """
    global _filters
    _filters = filters


def clear_search_filters():
    """Clear any active search filters."""
    global _filters
    _filters = None


@tool
def search_documents(query: str, k: int = 10) -> str:
    """
    Search for relevant educational documents and materials in the knowledge base.

    Use this tool FIRST before answering any questions to find accurate, relevant information.
    This tool searches through a database of educational content and returns the most relevant
    documents that can help you create accurate, fact-based lesson outlines.

    The search automatically filters by subject and grade level when specified in the request.
    It retrieves multiple documents to ensure comprehensive coverage of the topic.

    Args:
        query: The topic or question to search for (e.g., "environmental protection", "Vietnamese history")
        k: Number of documents to retrieve (recommended: 10 or more for better coverage)

    Returns:
        A formatted string containing the content of relevant documents
    """
    if _repository is None:
        return "Error: Knowledge base repository is not available."

    # Build filter dict for PGVector metadata filtering
    filter_dict = None
    if _filters:
        filter_dict = {}
        if _filters.get("subject_code"):
            filter_dict["subject_code"] = _filters["subject_code"]
        if _filters.get("grade"):
            filter_dict["grade"] = int(_filters["grade"])

    # Enforce minimum k value (LLM sometimes passes k=1 which is too few)
    k = max(k, 5)  # At least 5 documents

    print(
        f"[DEBUG] search_documents called with query: {query}, k: {k}, filters: {filter_dict}"
    )

    # Perform similarity search with filters
    docs = _repository.similarity_search(
        query=query, k=k, filter=filter_dict if filter_dict else None
    )

    # Format documents as a readable string
    if not docs:
        filter_info = ""
        if filter_dict:
            filter_info = f" with filters: {filter_dict}"
        return (
            f"No relevant documents found in the knowledge base{filter_info}."
        )

    result = []
    for i, doc in enumerate(docs, 1):
        result.append(f"Document {i}:")
        result.append(f"Content: {doc.page_content}")
        if doc.metadata:
            result.append(f"Metadata: {doc.metadata}")
        result.append("")  # Empty line for separation

    return "\n".join(result)


tools = [search_documents]

__all__ = [
    "tools",
    "set_global_repository",
    "set_search_filters",
    "clear_search_filters",
]
