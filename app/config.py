from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    endee_host: str = "http://localhost:8080"
    endee_auth_token: str = ""
    index_name: str = "insurance_policies"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    llm_model: str = "gpt-3.5-turbo"
    chunk_size: int = 500
    chunk_overlap: int = 50

    class Config:
        env_file = ".env"


settings = Settings()
