import os
from typing import List

from dotenv import load_dotenv
from langchain.chat_models.openai import ChatOpenAI
from langchain_core.messages import BaseMessage


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

    def run(self, model: str, messages: List[BaseMessage], **params) -> str:
        resp = self.client.invoke(
            input=messages, model=model, **params
        ).content

        if isinstance(resp, list):
            return " ".join(str(item) for item in resp)

        return resp

    async def stream(self, model: str, messages: List[BaseMessage], **params):
        resp = self.client.stream(input=messages, model=model, **params)

        for chunk in resp:
            yield chunk.content
