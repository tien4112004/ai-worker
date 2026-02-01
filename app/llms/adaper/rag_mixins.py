"""RAG Adapter Mixin for adding RAG capabilities to text models."""

from typing import Any, Dict, Iterator, List, Optional, Tuple

from langgraph.prebuilt.chat_agent_executor import create_tool_calling_executor

from app.llms.tool.agent_tools import tools
from app.schemas.token_usage import TokenUsage


class RAGAdapterMixin:
    """Mixin to add RAG capabilities to LLM adapters."""

    def run_rag(
        self,
        query: str,
        system_prompt: str,
        return_source_documents: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[Dict[str, Any], TokenUsage]:
        """
        Run RAG pipeline (Stuff Strategy).

        Args:
            query: User query
            system_prompt: System prompt for the agent
            return_source_documents: Whether to return source documents
            filters: Optional filters for document search (e.g., {"subject": "math", "grade": "5"})
            **kwargs: Additional parameters

        Returns:
            Tuple of (result dictionary, token usage)
        """
        from langchain_core.messages import HumanMessage

        from app.llms.tool.agent_tools import (
            clear_search_filters,
            set_search_filters,
        )

        if not hasattr(self, "client"):
            raise ValueError(
                "Adapter must have 'client' attribute to use RAGMixin"
            )

        try:
            # Set filters before creating the agent
            if filters:
                set_search_filters(filters)
                print(f"[DEBUG] RAG filters set: {filters}")

            # Define agent
            agent = create_tool_calling_executor(
                self.client,
                tools,
                prompt=system_prompt,
            )

            # Execute - LangGraph agents expect input with messages key
            response = agent.invoke(
                input={"messages": [HumanMessage(content=query)]}
            )

            print("[DEBUG] RAG Response: ", response)

            # Extract the final AI message from response
            messages = response.get("messages", [])
            final_message = messages[-1] if messages else None

            result = {
                "answer": final_message.content if final_message else "",
                "query": query,
            }

            if return_source_documents and "context" in response:
                result["source_documents"] = self._format_source_documents(
                    response["context"]
                )
                result["num_sources"] = len(response["context"])

            # Extract token usage from the final message if available
            token_usage = self._extract_token_usage(final_message)

            return result, token_usage

        finally:
            # Always clear filters after execution to avoid leaking to next request
            clear_search_filters()

    def stream_rag(
        self,
        query: str,
        system_prompt: str,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Iterator:
        """
        Stream RAG pipeline responses using tool-calling agent.

        Yields str content chunks as the LLM generates them, then
        yields a single TokenUsage object as the final item.

        Args:
            query: User query
            system_prompt: System prompt for the agent
            filters: Optional filters for document search

        Yields:
            str chunks of the LLM response, followed by a TokenUsage
        """
        from langchain_core.messages import AIMessageChunk, HumanMessage

        from app.llms.tool.agent_tools import (
            clear_search_filters,
            set_search_filters,
        )

        if not hasattr(self, "client"):
            raise ValueError(
                "Adapter must have 'client' attribute to use RAGMixin"
            )

        try:
            if filters:
                set_search_filters(filters)
                print(f"[DEBUG] RAG stream filters set: {filters}")

            agent = create_tool_calling_executor(
                self.client,
                tools,
                prompt=system_prompt,
            )

            last_ai_chunk = None
            for chunk, metadata in agent.stream(
                {"messages": [HumanMessage(content=query)]},
                stream_mode="messages",
            ):
                if not isinstance(chunk, AIMessageChunk):
                    continue
                # Skip tool-use chunks (agent calling search_documents)
                if chunk.tool_calls:
                    continue
                if chunk.content and isinstance(chunk.content, str):
                    last_ai_chunk = chunk
                    yield chunk.content

            # Yield token usage as the final item
            yield self._extract_token_usage(last_ai_chunk)

        finally:
            clear_search_filters()

    def _format_source_documents(
        self, source_documents: List[Any]
    ) -> List[Dict[str, Any]]:
        """Format source documents for response."""
        return [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in source_documents
        ]

    def _extract_token_usage(self, message: Any) -> TokenUsage:
        """Extract token usage from AI message response_metadata."""
        if message and hasattr(message, "response_metadata"):
            metadata = message.response_metadata
            usage_metadata = metadata.get("usage_metadata", {})

            input_tokens = usage_metadata.get("prompt_token_count", 0)
            output_tokens = usage_metadata.get("candidates_token_count", 0)
            total_tokens = usage_metadata.get(
                "total_token_count", input_tokens + output_tokens
            )

            return TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                model=getattr(self, "model_name", "unknown"),
                provider=getattr(self, "provider", "unknown"),
            )

        # Return zero usage if no metadata available
        return TokenUsage(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            model=getattr(self, "model_name", "unknown"),
            provider=getattr(self, "provider", "unknown"),
        )
