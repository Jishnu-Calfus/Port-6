from functools import lru_cache

from langchain_chroma import Chroma
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_openai import OpenAIEmbeddings

from port6.caching.embedding_cache import get_cached_embeddings
from port6.config import settings
from port6.ingestion.chunker import get_child_splitter


@lru_cache
def get_embeddings() -> CacheBackedEmbeddings:
    """Redis-cached OpenAI embedder — identical text (a repeat ingestion, or
    a repeated query) is never sent to the OpenAI API twice."""
    underlying = OpenAIEmbeddings(model=settings.openai_embedding_model, api_key=settings.openai_api_key)
    return get_cached_embeddings(underlying, model_name=settings.openai_embedding_model)


@lru_cache
def get_chroma_store() -> Chroma:
    """The one production vector store. persist_directory is the literal
    answer to 'show me where the embeddings are stored' — vectors, child
    chunk text, and metadata (document_id, version, active, page_number) all
    live together on disk here, not split across separate files."""
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


@lru_cache
def get_parent_docstore():
    """Parent chunks (the full-context text handed to the LLM) live here,
    keyed by the same id ParentDocumentRetriever stamps onto their child
    chunks in Chroma — that id is how a matched child chunk gets expanded
    back to its parent at retrieval time."""
    return create_kv_docstore(LocalFileStore(settings.parent_store_dir))


@lru_cache
def get_parent_document_retriever() -> ParentDocumentRetriever:
    """parent_splitter=None is deliberate: chunker.split_into_parents()
    already did the page-aware parent splitting (File 5). This only needs to
    take those parents and split each one into children for embedding."""
    return ParentDocumentRetriever(
        vectorstore=get_chroma_store(),
        docstore=get_parent_docstore(),
        child_splitter=get_child_splitter(),
        parent_splitter=None,
    )
