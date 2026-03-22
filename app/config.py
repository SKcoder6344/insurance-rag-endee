"""
Central configuration — all values loaded from environment variables.
Never hardcode secrets here.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    LLM_MAX_TOKENS: int = 800
    LLM_TEMPERATURE: float = 0.1

    # Endee Vector DB
    ENDEE_URL: str = "http://localhost:8080"
    ENDEE_AUTH_TOKEN: str = ""
    INDEX_NAME: str = "insurance_policies_hybrid"

    # Embeddings
    DENSE_MODEL: str = "all-MiniLM-L6-v2"
    DENSE_DIM: int = 384

    # Retrieval
    TOP_K: int = 5

    # Observability
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton — parsed once at startup."""
    return Settings()


settings = get_settings()
