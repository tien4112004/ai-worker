from typing import Optional

from pydantic import BaseModel


class TokenUsage(BaseModel):
    """Schema for tracking token usage in LLM requests."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: Optional[str] = None
    provider: Optional[str] = None

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage objects together."""
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )
