from contextlib import asynccontextmanager

from fastapi import FastAPI

from port6.api.routers import ingest, query
from port6.caching.llm_cache import configure_llm_cache
from port6.observability.tracing import configure_tracing


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_tracing()
    configure_llm_cache()
    yield


app = FastAPI(title="Port6 Document RAG Assistant", lifespan=lifespan)
app.include_router(ingest.router)
app.include_router(query.router)
