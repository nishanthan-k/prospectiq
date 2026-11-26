from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    gemini_api_key: str = Field(default="", description="Google Gemini API Key")
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        description="Google AI Studio OpenAI-compatible base URL",
    )
    default_model: str = "gemini-3.1-flash-lite"
    fast_model: str = "gemini-3.1-flash-lite"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    log_format: str = "text"
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0)
    llm_retry_min_wait: float = Field(default=1.0, ge=0)
    llm_retry_max_wait: float = Field(default=10.0, ge=0)
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/prospectiq",
        description="Async SQLAlchemy database URL (PostgreSQL asyncpg)",
    )
    db_echo: bool = Field(default=False, description="Enable SQLAlchemy SQL query logging")

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
