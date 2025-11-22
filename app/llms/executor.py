from typing import Any, Dict

from app.llms.adaper.image_models.imagen import ImagenAdapter
from app.llms.adaper.text_models.gemini import GeminiAdapter
from app.llms.adaper.text_models.localai import LocalAIAdapter
from app.llms.adaper.text_models.open_router import OpenRouterAdapter
from app.llms.adaper.text_models.openai import OpenAIAdapter


class LLMExecutor:
    def __init__(self) -> None:
        self.adapters = {
            "openai": OpenAIAdapter,
            "google": GeminiAdapter,
            "openrouter": OpenRouterAdapter,
            "localai": LocalAIAdapter,
        }
        self.image_adapters = {"google": ImagenAdapter}

    def _adapter(self, provider: str):
        if provider in self.adapters:
            return self.adapters[provider]

        raise ValueError(f"Unknown provider: {provider}")

    def _image_adapter(self, provider: str):
        if provider in self.image_adapters:
            return self.image_adapters[provider]

        raise ValueError(f"Unknown image provider: {provider}")

    def batch(self, provider: str, model: str, messages, **params) -> str:
        adapter_class = self._adapter(provider)
        adapter = adapter_class(model_name=model)
        return adapter.run(model=model, messages=messages, **params)

    def stream(self, provider: str, model: str, messages, **params):
        adapter_class = self._adapter(provider)
        adapter = adapter_class(model_name=model)
        for chunk in adapter.stream(model=model, messages=messages, **params):
            yield chunk

    def generate_image(
        self, provider: str, model: str, message: str, **params
    ) -> Dict[str, Any]:
        print("Generating image with model:", model)
        adapter_class = self._image_adapter(provider)
        if adapter_class is None:
            raise ValueError(f"Image adapter for {provider} is not available")

        adapter = adapter_class(model=model)
        return adapter.generate(message=message, **params)
