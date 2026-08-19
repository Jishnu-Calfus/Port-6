from redis import Redis

from port6.caching.redis_client import get_redis_client
from port6.retrieval.vectorstore import get_chroma_store


def redis_dependency() -> Redis:
    return get_redis_client()


def chroma_dependency():
    return get_chroma_store()
