"""Settings loaded from environment variables / .env."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("../../data")
    models_dir: Path = Path("../../models")
    database_url: str = "sqlite:///./flow_er.db"
    anthropic_api_key: str | None = None


settings = Settings()
