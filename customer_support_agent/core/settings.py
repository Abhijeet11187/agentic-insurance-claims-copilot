from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# settings.py -> core -> customer_support_agent -> project root
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "Insurance Claims Support Copilot"
    app_env: str = "local"
    api_base_url: str = "http://localhost:8000"

    # --- LLM (OpenAI) ---
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    llm_temperature: float = 0.2
    llm_timeout: int = 60
    llm_max_retries: int = 3

    # --- Embeddings (OpenAI, same key) ---
    embedding_model: str = "text-embedding-3-small"
    embedding_dims: int = 1536
    embedding_batch_size: int = 64
    enable_semantic: bool = True

    # --- Paths ---
    data_dir: Path = BASE_DIR / "data"
    db_path: Path = BASE_DIR / "data" / "app.db"
    chroma_path: Path = BASE_DIR / "data" / "chroma"
    knowledge_base_dir: Path = BASE_DIR / "knowledge_base"

    # --- RAG ---
    chunk_size: int = 900
    chunk_overlap: int = 150
    rag_top_k: int = 4

    # --- Memory ---
    memory_top_k: int = 4

    @property
    def semantic_enabled(self) -> bool:
        """Embeddings use the same OpenAI key as the chat model."""
        return self.enable_semantic and bool(self.openai_api_key.strip())

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings