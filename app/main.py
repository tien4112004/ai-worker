from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api
from app.core.config import settings
from app.llms.executor import LLMExecutor
from app.prompts.loader import PromptStore
from app.services.content_service import ContentService


@asynccontextmanager
async def lifespan(app: FastAPI):
    prompt_store = PromptStore()
    llm_executor = LLMExecutor()
    content_service = ContentService(
        llm_executor=llm_executor, prompt_store=prompt_store
    )

    app.state.settings = settings
    app.state.content_service = content_service

    yield
    # Shutdown code can go here


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
