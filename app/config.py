"""One configuration source for API, jobs, and migration commands."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")
    database_url: str = "sqlite+aiosqlite:///./app.db"
    auto_create_tables: bool = True
    llm_enabled: bool = False
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_retries: int = Field(default=1, ge=0, le=3)
    web_search_provider: Literal["demo", "disabled", "brave", "searxng"] = "demo"
    brave_api_key: SecretStr | None = None
    searxng_url: str = "http://localhost:8080"
    history_path: Path = ROOT / "data/history.csv"
    demo_sources_path: Path = ROOT / "data/demo_sources.json"
    max_search_iterations: int = Field(default=2, ge=1, le=3)
    search_result_limit: int = Field(default=10, ge=1, le=50)
    max_pages_per_lot: int = Field(default=20, ge=1, le=100)
    max_queries_per_iteration: int = Field(default=3, ge=1, le=10)
    request_timeout: float = Field(default=15, gt=0, le=120)
    ollama_timeout: float = Field(default=60, gt=0, le=300)
    max_page_bytes: int = Field(default=2_000_000, ge=1000)
    max_upload_bytes: int = Field(default=5_000_000, ge=1000)
    candidate_text_min_similarity: float = Field(default=0.08, gt=0, le=1)
    unknown_ratio_threshold: float = Field(default=0.5, ge=0, le=1)
    confirmed_weight: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    contradicted_weight: float = Field(default=-2.0, lt=0, allow_inf_nan=False)
    unknown_weight: float = Field(default=0.0, allow_inf_nan=False)
    history_win_bonus: float = Field(default=2.0, ge=0, le=10, allow_inf_nan=False)
    max_history_bonus: float = Field(default=5.0, ge=0, le=10, allow_inf_nan=False)
    okpd2_min_similarity: float = Field(default=0.45, ge=0, le=1)
    okpd2_min_margin: float = Field(default=0.05, ge=0, le=1)
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    log_level: str = "INFO"

    @model_validator(mode="after")
    def check_weights(self):
        if not self.contradicted_weight < self.unknown_weight < self.confirmed_weight:
            raise ValueError("Weights must satisfy contradicted < unknown < confirmed")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
