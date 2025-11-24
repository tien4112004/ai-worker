import json
import logging
import os
from typing import ClassVar, List

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    app_name: str = "fastapi-starter"
    default_model: str = "gemini-2.5-flash-lite"
    base_url: str = "http://localhost:8080"

    # LLM Configuration
    google_api_key: str = os.getenv(
        "GOOGLE_API_KEY", ""
    )  # Alternative name for Gemini API key
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # VertexAI Configuration
    service_account_json: str = os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS", "./service_account.json"
    )
    project_id: str = os.getenv("VERTEX_PROJECT_ID", "")
    location: str = os.getenv("VERTEX_LOCATION", "us-central1")

    max_retries: int = int(os.getenv("MAX_RETRIES", 3))

    # Default LLM parameters
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", 0.7))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", 2048))

    # CORS Configuration
    allowed_origins: str = os.getenv("ALLOWED_ORIGINS", "*")
    allowed_methods: str = os.getenv(
        "ALLOWED_METHODS", "GET,POST,PUT,DELETE,OPTIONS"
    )
    allowed_headers: str = os.getenv("ALLOWED_HEADERS", "*")
    allow_credentials: bool = (
        os.getenv("ALLOWED_CREDENTIALS", "True").lower() == "true"
    )

    # LocalAI Configuration
    localai_base_url: str = os.getenv(
        "LOCALAI_BASE_URL", "http://localhost:8083"
    )
    localai_api_key: str = os.getenv("LOCALAI_API_KEY", "sk-local")

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="", extra="allow"
    )

    logger: ClassVar[logging.Logger] = logging.getLogger("uvicorn.error")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

settings = Settings()
