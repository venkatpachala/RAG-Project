"""
config.py — Centralized application settings
Uses pydantic-settings to read from .env, validate types, and provide defaults.

Usage:
    from src.core.config import settings
    print(settings.GOOGLE_API_KEY)
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """
    BaseSettings automatically reads from environment variables and .env files.
    Field(...) means required — no default, app won't start without it.
    Field("default") means optional — falls back to the given value.
    """

    # Google Gemini
    GOOGLE_API_KEY: str = Field(..., description="Your Google Gemini API key")
    GEMINI_MODEL: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model for answer generation"
    )

    # Embeddings
    EMBEDDING_MODEL: str = Field(
        default="fast",
        description="Embedding model preset: 'fast', 'balanced', or 'best'"
    )

    # Vector store
    CHROMA_PERSIST_DIR: str = Field(
        default="./vector_db",
        description="Local directory where ChromaDB saves its data"
    )
    COLLECTION_NAME: str = Field(
        default="knowledge_base",
        description="ChromaDB collection name"
    )

    # RAG Tuning
    RETRIEVAL_TOP_K: int = Field(
        default=5,
        description="Number of chunks retrieved per query"
    )
    CHUNK_SIZE: int = Field(
        default=800,
        description="Characters per document chunk"
    )
    CHUNK_OVERLAP: int = Field(
        default=150,
        description="Overlap between adjacent chunks"
    )

    # Data directories
    DATA_DIR: str = Field(
        default="./data",
        description="Root directory for all generated data"
    )

    # File Uploads
    UPLOAD_DIR: str = Field(default="./uploads")
    MAX_FILE_SIZE_MB: int = Field(default=50)

    # Pydantic settings config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    @property
    def chunks_dir(self) -> Path:
        return Path(self.DATA_DIR) / "chunks"

    @property
    def embeddings_dir(self) -> Path:
        return Path(self.DATA_DIR) / "embeddings"

    @property
    def logs_dir(self) -> Path:
        return Path(self.DATA_DIR) / "logs"

    @property
    def visualizations_dir(self) -> Path:
        return Path(self.DATA_DIR) / "visualizations"

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    def ensure_directories(self):
        """Create all necessary directories"""
        for d in [self.chunks_dir, self.embeddings_dir, self.logs_dir, 
                  self.visualizations_dir, Path(self.UPLOAD_DIR)]:
            d.mkdir(parents=True, exist_ok=True)


# Singleton instance
settings = Settings()
settings.ensure_directories()