from langchain_core.globals import set_llm_cache
from langchain_redis import RedisCache

from port6.config import settings


def configure_llm_cache() -> None:
    """Called once at app startup. After this, every ChatOpenAI call in the
    process — generation, and the intent classifier — transparently checks
    Redis first, keyed on the exact rendered prompt (system prompt + the
    live retrieved context + the question), before hitting the API.

    Note: langchain_redis.RedisCache emits a PendingDeprecationWarning about
    its internal deserialization not yet taking an explicit allowed_objects
    allowlist. That's a third-party internal (no constructor param exposes
    it), and the practical risk is low here — Redis isn't exposed publicly,
    and the only writer to these keys is this same trusted process."""
    set_llm_cache(RedisCache(redis_url=settings.redis_url, ttl=settings.llm_cache_ttl_seconds))
