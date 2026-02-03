import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import aiplatform
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register
from rich.repr import auto

from app.api.router import api
from app.core.config import settings
from app.llms.executor import LLMExecutor
from app.middleware import injectCustomTraceId
from app.prompts.loader import PromptStore
from app.services.content_service import ContentService
from app.services.exam_service import ExamService


@asynccontextmanager
async def lifespan(app: FastAPI):

    llm_tracer = register(
        project_name=settings.phoenix_project_name,
        endpoint=settings.phoenix_collector_endpoint,
        auto_instrument=False,  # Disabled to prevent duplicate spans
    )

    # Only instrument LangChain to avoid duplicate spans from both LangChain and underlying GoogleGenAI client
    LangChainInstrumentor().instrument(tracer_provider=llm_tracer)
    GoogleGenAIInstrumentor().instrument(tracer_provider=llm_tracer)

    prompt_store = PromptStore()
    llm_executor = LLMExecutor()
    content_service = ContentService(
        llm_executor=llm_executor,
        prompt_store=prompt_store,
    )
    exam_service = ExamService(
        llm_executor=llm_executor, prompt_store=prompt_store
    )

    app.state.settings = settings
    app.state.content_service = content_service
    app.state.exam_service = exam_service

    def init_vertexai():
        """Initialize Vertex AI settings."""
        import os
        import vertexai
        from google.oauth2 import service_account

        # Skip initialization if service account file doesn't exist (for testing/mock mode)
        if not os.path.exists(settings.service_account_json):
            print(f"Warning: Service account file not found at {settings.service_account_json}")
            print("Vertex AI initialization skipped - using mock mode")
            return

        service_account_info = json.load(open(settings.service_account_json))
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info
        )

        aiplatform.init(
            project=settings.project_id,
            location=settings.location,
            credentials=credentials,
        )

    init_vertexai()

    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"},
        lifespan=lifespan,
    )

    app.middleware("http")(injectCustomTraceId)
    app.include_router(api, prefix="/api")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=settings.allow_credentials,
        allow_methods=settings.allowed_methods,
        allow_headers=settings.allowed_headers,
    )

    return app


app = create_app()
