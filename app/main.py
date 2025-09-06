from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.router import api
from app.llms.factory import LLMFactory
from app.llms.service import LLMService
from app.services.content_service import ContentService

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code can go here
    llm_factory = LLMFactory({
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens
    })
    llm_service = LLMService(llm_factory=llm_factory)
    content_service = ContentService(model_name=settings.default_model, llm_service=llm_service)
    
    app.state.settings = settings
    app.state.content_service = content_service

    yield
    # Shutdown code can go here

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"},
        lifespan=lifespan
    )
    
    app.include_router(api, prefix="/api/v1")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=settings.allow_credentials,
        allow_methods=settings.allowed_methods,
        allow_headers=settings.allowed_headers,
    )

    return app

app = create_app()