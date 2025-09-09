from app.llms.adaper.text_models.gemini import GeminiAdapter
from app.llms.adaper.text_models.openai import OpenAIAdapter


class LLMExecutor:
    def __init__(self) -> None:
        self.adapters = {"openai": OpenAIAdapter, "gemini": GeminiAdapter}

    def _adapter(self, provider: str):
        if provider in self.adapters:
            return self.adapters[provider]

        raise ValueError(f"Unknown provider: {provider}")

    def batch(self, provider: str, model: str, messages, **params) -> str:
        adapter_class = self._adapter(provider)
        adapter = adapter_class(model_name=model)
        return adapter.run(model=model, messages=messages, **params)

    def stream(self, provider: str, model: str, messages, **params):
        adapter_class = self._adapter(provider)
        adapter = adapter_class(model_name=model)
        for chunk in adapter.stream(model=model, messages=messages, **params):
            yield chunk
