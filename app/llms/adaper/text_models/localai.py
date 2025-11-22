import os
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI


class LocalAIAdapter:
    def __init__(self, model_name: str, **params):
        load_dotenv()

        localai_base_url = os.getenv(
            "LOCALAI_BASE_URL", "http://localhost:8083"
        )
        localai_api_key = os.getenv("LOCALAI_API_KEY", "sk-local")

        if not localai_base_url:
            raise ValueError(
                "LOCALAI_BASE_URL must be set in env or defaults to http://localhost:8083"
            )

        # Initialize the ChatOpenAI client with LocalAI endpoint
        # LocalAI uses OpenAI-compatible API at /v1 endpoint
        self.client = ChatOpenAI(
            model=model_name,
            base_url=f"{localai_base_url}/v1",
            api_key=localai_api_key,
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
