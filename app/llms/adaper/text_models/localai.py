from typing import List

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings


class LocalAIAdapter:
    def __init__(self, model_name: str, **params):
        self.client = ChatOpenAI(
            model=model_name,
            base_url=f"{settings.localai_base_url}/v1",
            api_key=settings.localai_api_key,
            **params,
        )

    def run(self, model: str, messages: List[BaseMessage], **params) -> str:
        resp = self.client.invoke(input=messages, **params).content

        if isinstance(resp, list):
            return " ".join(str(item) for item in resp)

        return resp

    def stream(self, model: str, messages: List[BaseMessage], **params):
        resp = self.client.stream(input=messages, **params)

        for chunk in resp:
            yield chunk.content
