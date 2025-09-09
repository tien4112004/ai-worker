from typing import List

from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI


class GeminiAdapter:
    def __init__(self, model_name: str, **params):
        # REQUIRES: GOOGLE_API_KEY in env
        self.client = ChatGoogleGenerativeAI(
            model=model_name, **params, convert_system_message_to_human=True
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
