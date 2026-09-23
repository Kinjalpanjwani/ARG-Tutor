from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    app_name: str = "AI Tutor API"
    app_env: str = "development"
    api_prefix: str = "/api"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    groq_api_key: str | None = Field(default=None, repr=False)
    groq_model: str = "openai/gpt-oss-120b"
    light_groq_model: str = "llama-3.1-8b-instant"
    groq_whisper_model: str = "whisper-large-v3"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 6
    retrieval_threshold: float = 0.25
    max_upload_mb: int = 25
    language_detection_enabled: bool = False
    tts_voice_english: str = "en-US-AriaNeural"
    tts_voice_urdu: str = "ur-PK-AsadNeural"
    data_dir: Path = BACKEND_DIR / "data"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def vector_store_dir(self) -> Path:
        return self.data_dir / "vector_store"

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
