from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    groq_api_key: str
    default_model: str = "openai/gpt-oss-120b"
    fast_model: str = "openai/gpt-oss-20b"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    log_format: str = "text"
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0)
    llm_retry_min_wait: float = Field(default=1.0, ge=0)
    llm_retry_max_wait: float = Field(default=10.0, ge=0)

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
