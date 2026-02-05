import os
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import BaseMessage

from app.schemas.token_usage import TokenUsage


class OpenRouterAdapter:
    def __init__(self, **params):
        load_dotenv()

        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        openrouter_base_url = os.getenv("OPENROUTER_BASE_URL")

        if not openrouter_api_key and not openrouter_base_url:
            raise ValueError(
                "OPENROUTER_API_KEY and OPENROUTER_BASE_URL must be set in env or passed as params"
            )

        params["openrouter_api_key"] = openrouter_api_key
        params["openrouter_api_base"] = openrouter_base_url

        self.client = ChatOpenAI(
            temperature=0.7,
            api_key=params.get("openrouter_api_key"),
            base_url=params.get("openrouter_api_base"),
        )

    def run(
        self, model: str, messages: List[BaseMessage], **params
    ) -> Tuple[str, TokenUsage]:
        resp = self.client.invoke(input=messages, model=model, **params)
        content = resp.content

        if isinstance(content, list):
            content = " ".join(str(item) for item in content)

        # Extract token usage
        usage = TokenUsage()
        if hasattr(resp, "usage_metadata") and resp.usage_metadata:
            usage = TokenUsage(
                input_tokens=resp.usage_metadata.get("input_tokens", 0),
                output_tokens=resp.usage_metadata.get("output_tokens", 0),
                total_tokens=resp.usage_metadata.get("input_tokens", 0)
                + resp.usage_metadata.get("output_tokens", 0),
                model=model,
                provider="openrouter",
            )

        return content, usage

    def stream(
        self, model: str, messages: List[BaseMessage], **params
    ) -> Tuple[List[str], TokenUsage]:
        """Stream response and collect token usage.

        Returns a tuple of (chunks, token_usage) where chunks is a list of content chunks.
        """
        chunks = []
        total_input_tokens = 0
        total_output_tokens = 0

        resp_stream = self.client.stream(input=messages, model=model, **params)

        for chunk in resp_stream:
            if chunk.content:
                chunks.append(chunk.content)
            # Sum token usage from each chunk
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                total_input_tokens += chunk.usage_metadata.get(
                    "input_tokens", 0
                )
                total_output_tokens += chunk.usage_metadata.get(
                    "output_tokens", 0
                )

        total = total_input_tokens + total_output_tokens
        usage = TokenUsage(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            total_tokens=total,
            model=model,
            provider="openrouter",
        )

        return chunks, usage
