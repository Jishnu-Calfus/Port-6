from functools import lru_cache

import redis

from port6.config import settings


@lru_cache
def get_redis_client() -> redis.Redis:
    """Single shared connection every caching use (embedding cache, LLM
    cache, ingestion dedup lock) reads from the same place — no separately
    configured connections drifting out of sync."""
    return redis.Redis.from_url(settings.redis_url, decode_responses=False)
