from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "documents"

    documents_dir: str = "./data/documents"
    parent_store_dir: str = "./data/parent_store"

    parent_chunk_size: int = 1500
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 400
    child_chunk_overlap: int = 60

    retrieval_top_k: int = 20
    rerank_top_n: int = 5
    bm25_weight: float = 0.4
    dense_weight: float = 0.6
    reranker_model: str = "BAAI/bge-reranker-base"

    ocr_language: str = "eng"

    api_base_url: str = "http://localhost:8000"

    redis_url: str = "redis://localhost:6379/0"
    llm_cache_ttl_seconds: int = 60 * 60 * 24  # 1 day

    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "port6-rag"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
