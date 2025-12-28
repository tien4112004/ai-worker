import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import aiplatform

from app.api.router import api
from app.core.config import settings
from app.llms.executor import LLMExecutor
from app.prompts.loader import PromptStore
from app.services.content_service import ContentService
from app.services.exam_service import ExamService


@asynccontextmanager
async def lifespan(app: FastAPI):
    prompt_store = PromptStore()
    llm_executor = LLMExecutor()
    content_service = ContentService(
        llm_executor=llm_executor, prompt_store=prompt_store
    )
    exam_service = ExamService(
        llm_executor=llm_executor, prompt_store=prompt_store
    )

    app.state.settings = settings
    app.state.content_service = content_service
    app.state.exam_service = exam_service

    def init_vertexai():
        """Initialize Vertex AI settings."""
        import vertexai
        from google.oauth2 import service_account

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
