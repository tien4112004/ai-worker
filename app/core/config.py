import logging
from typing import ClassVar, List
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

import os

load_dotenv()

class Settings(BaseSettings):
    app_name: str = "fastapi-starter"
    default_model: str = "gemini-2.5-flash-lite"
    base_url : str = "http://localhost:8080"
    
    # LLM Configuration
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")  # Alternative name for Gemini API key
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Default LLM parameters
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", 0.7))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", 2048))

    # CORS Configuration
    allowed_origins: str = os.getenv('ALLOWED_ORIGINS', '*')
    allowed_methods: str = os.getenv('ALLOWED_METHODS', 'GET,POST,PUT,DELETE,OPTIONS')
    allowed_headers: str = os.getenv('ALLOWED_HEADERS', '*')
    allow_credentials: bool = os.getenv('ALLOWED_CREDENTIALS', 'True').lower() == 'true'
    print(f"Allowed Origins: {allowed_origins}")
    print(f"Allowed Methods: {allowed_methods}")
    print(f"Allowed Headers: {allowed_headers}")
    print(f"Allow Credentials: {allow_credentials}")

    logger: ClassVar[logging.Logger] = logging.getLogger("uvicorn.error")

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="allow")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

settings = Settings()