"""Settings loaded from environment variables / .env."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Data and models live next to the repository, never inside it: <repo>/../data, <repo>/../models.
_OUTSIDE_REPO = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = _OUTSIDE_REPO / "data"
    models_dir: Path = _OUTSIDE_REPO / "models"
    database_url: str = "sqlite:///./flow_er.db"
    anthropic_api_key: str | None = None

    @property
    def splits_dir(self) -> Path:
        """Registry and frozen splits: they name donors, so they stay out of git."""
        return self.data_dir / "_splits"


settings = Settings()
