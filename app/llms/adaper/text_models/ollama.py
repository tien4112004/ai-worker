import os
from typing import List

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import BaseMessage


class OllamaAdapter:
    def __init__(self, model_name: str, **params):
        load_dotenv()

        ollama_base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

        if not ollama_base_url:
            raise ValueError(
                "OLLAMA_URL must be set in env or defaults to http://localhost:11434"
            )

        # Initialize the ChatOllama client
        self.client = ChatOllama(
            model=model_name, base_url=ollama_base_url, **params
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
