from typing import Any, Dict, Iterable, List

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI


class OpenAIAdapter:
    def __init__(self, model_name: str, **params):
        # REQUIRES: OPENAI_API_KEY in env
        self.client = ChatOpenAI(model=model_name, **params)

    def run(self, model: str, messages: List[BaseMessage], **params) -> str:
        resp = self.client.batch(model=model, messages=messages, **params)

        if isinstance(resp, list):
            return " ".join(str(item) for item in resp)

        return resp

    async def stream(self, model: str, messages: List[BaseMessage], **params):
        resp = self.client.stream(model=model, messages=messages, **params)

        for chunk in resp:
            yield chunk.content
