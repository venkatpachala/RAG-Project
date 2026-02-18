# """
# config.py — Centralized application settings

# Uses pydantic-settings to:
#   1. Read values from the .env file automatically
#   2. Validate types (e.g., PORT must be an int)
#   3. Provide defaults so the app doesn't crash if a var is missing

# Usage anywhere in the codebase:
#     from config import settings
#     print(settings.NVIDIA_API_KEY)
# """

#pydanyic-setting -> Basesettings inherited class which does the above funtions
#Instead of calling os.getenv stuff

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
import os


class Settings(BaseSettings):
    """
    BaseSettings automatically reads from environment variables and .env files.
    Field(...) means required — no default, app won't start without it.
    Field("default") means optional — falls back to the given value.
    """

    NVIDIA_API_KEY: str = Field(..., description="Your NVIDIA API key from build.nvidia.com")
    NVIDIA_BASE_URL: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="NVIDIA's OpenAI-compatible API base URL"
    )
    NVIDIA_LLM_MODEL: str = Field(
        default="meta/llama-3.1-70b-instruct",
        description="Model used for answer generation"
    )
    NVIDIA_EMBEDDING_MODEL: str = Field(
        default="nvidia/nv-embedqa-e5-v5",
        description="Model used for creating text embeddings"
    )

    # Vector store
    CHROMA_PERSIST_DIR: str = Field(
        default="./chroma_db",
        description="Local directory where ChromaDB saves its data"
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
        description="Overlap between adjacent chunks to avoid cutting context"
    )

    #  Server
    BACKEND_HOST: str = Field(default="0.0.0.0")
    BACKEND_PORT: int = Field(default=8000)
    ALLOWED_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:5173"
    )

    # Filed Uploads
    UPLOAD_DIR: str = Field(default="./uploads")
    MAX_FILE_SIZE_MB: int = Field(default=50)

    # Pydantic settings config
    # This tells pydantic-settings WHERE to look for the .env file
    model_config = SettingsConfigDict(
        env_file=".env",           # Read from .env in project root
        env_file_encoding="utf-8",
        case_sensitive=True,       # NVIDIA_API_KEY != nvidia_api_key
        extra="ignore"             # Don't crash on unknown env vars
    )

    @property
    def allowed_origins_list(self) -> list[str]:
        """Convert comma-separated origins string → Python list for FastAPI CORS"""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]

    @property
    def max_file_size_bytes(self) -> int:
        """Convert MB → bytes for file size validation"""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    def ensure_directories(self):
        """Create necessary directories if they don't exist yet"""
        Path(self.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)


# Singleton instats

settings = Settings()