from typing import Optional

from app.schemas.token_usage import TokenUsage


class TokenTracker:
    """Utility class to track and aggregate token usage across requests."""

    def __init__(self):
        self.total_usage = TokenUsage()
        self.request_usages = []

    def add_usage(self, usage: TokenUsage) -> None:
        """Add token usage from a single request."""
        if isinstance(usage, dict):
            usage = TokenUsage(**usage)
        self.request_usages.append(usage)
        self.total_usage = self.total_usage + usage

    def get_total(self) -> TokenUsage:
        """Get total token usage."""
        return self.total_usage

    def get_usage_count(self) -> int:
        """Get number of tracked requests."""
        return len(self.request_usages)

    def reset(self) -> None:
        """Reset tracker."""
        self.total_usage = TokenUsage()
        self.request_usages = []

    def to_dict(self) -> dict:
        """Convert usage to dictionary."""
        return {
            "input_tokens": self.total_usage.input_tokens,
            "output_tokens": self.total_usage.output_tokens,
            "total_tokens": self.total_usage.total_tokens,
            "requests_count": len(self.request_usages),
        }
