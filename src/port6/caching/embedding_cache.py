from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_community.storage import RedisStore
from langchain_core.embeddings import Embeddings

from port6.config import settings

_NAMESPACE = "port6:embed"


def get_cached_embeddings(underlying: Embeddings, model_name: str) -> CacheBackedEmbeddings:
    """Wraps an embedder so identical text is never re-embedded — a repeat
    ingestion of the same chunk, or a repeated query, hits Redis instead of
    calling OpenAI again. Namespaced by model name so switching embedding
    models later can't silently return stale vectors from a different
    model's dimensions. query_embedding_cache=True also caches repeated
    queries, not just document chunks — useful across eval runs that ask the
    same golden-set questions many times."""
    store = RedisStore(redis_url=settings.redis_url)
    return CacheBackedEmbeddings.from_bytes_store(
        underlying,
        store,
        namespace=f"{_NAMESPACE}:{model_name}",
        query_embedding_cache=True,
        key_encoder="sha256",
    )
