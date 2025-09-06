# app/llms/factory.py
import os
from typing import Optional, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

class LLMFactory:
    def __init__(self, defaults: Optional[Dict[str, Any]] = None):
        self.defaults = defaults or {}

    def build(self, provider: str, model: str, *, params: Optional[Dict[str, Any]] = None) -> BaseChatModel:
        """Return a LangChain ChatModel for the requested provider+model."""
        p = (params or {}) | self.defaults

        if provider.lower() == "openai":
            # expects OPENAI_API_KEY in env
            return ChatOpenAI(model=model, **p)
        elif provider.lower() == "google":
            # expects GOOGLE_API_KEY in env
            return ChatGoogleGenerativeAI(model=model, **p, convert_system_message_to_human=True)
        elif provider.lower() == "anthropic":
            # expects ANTHROPIC_API_KEY in env
            return ChatAnthropic(model_name=model, **p)

        raise ValueError(f"Unsupported provider: {provider}")

    def build_from_model_string(self, model_string: str, *, params: Optional[Dict[str, Any]] = None) -> BaseChatModel:
        """Build LLM from a model string like 'gemini-2.5-flash-lite' or 'openai-gpt-4'."""
        # Parse model string to extract provider and model
        if model_string.startswith("gemini"):
            return self.build("google", model_string, params=params)
        elif model_string.startswith("gpt"):
            return self.build("openai", model_string, params=params)
        elif model_string.startswith("claude"):
            return self.build("anthropic", model_string, params=params)
        else:
            # Default to google if no clear provider prefix
            return self.build("google", model_string, params=params)
