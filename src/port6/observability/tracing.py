import os

from port6.config import settings


def configure_tracing() -> None:
    """Called once at app startup. LangSmith reads its config from plain OS
    environment variables (LANGSMITH_TRACING/LANGSMITH_PROJECT/
    LANGSMITH_API_KEY), not from our pydantic Settings object — same reason
    the OpenAI clients needed an explicit api_key= earlier. Disabled by
    default; when off, this is a no-op and nothing gets traced."""
    if not settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "false"
        return

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
