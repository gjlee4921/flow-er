"""Settings loaded from environment variables / .env."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("../../data")
    models_dir: Path = Path("../../models")
    # One folder per experiment: plan, gates, views, journal, chat. Kept outside the repo.
    workspace_dir: Path = Path("../../workspaces")
    database_url: str = "sqlite:///./flow_er.db"
    anthropic_api_key: str | None = None
    # The web app's agent runs on OpenAI for now; Anthropic slots in behind the same adapter.
    llm_provider: str = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.5"
    # How many tool rounds the agent may take in one turn before it must answer.
    agent_max_rounds: int = 10
    # Folders on this machine the start page may link to and read in place (comma-separated).
    import_roots: str = "~"
    # The largest file the start page accepts, in MB.
    upload_max_mb: int = 4096


settings = Settings()
