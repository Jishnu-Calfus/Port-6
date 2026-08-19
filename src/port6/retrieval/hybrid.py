import re

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from port6.config import settings
from port6.retrieval.vectorstore import get_chroma_store

ACTIVE_FILTER = {"active": True}

# BM25Retriever's default preprocessing is just text.split() — no
# punctuation stripping, so "(SOP-114)." and a query of "SOP-114" share zero
# tokens and never match. This keeps hyphenated codes intact as one token
# (via the repeated "-alnum" group) while still stripping surrounding
# punctuation, which is exactly the case BM25 exists to catch.
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


def active_child_documents() -> list[Document]:
    """Every active child chunk currently in Chroma, reconstructed as
    Documents with their real Chroma id copied into metadata['chunk_id'] —
    this is what the BM25 index gets built over."""
    raw = get_chroma_store()._collection.get(where=ACTIVE_FILTER, include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata={**metadata, "chunk_id": chunk_id})
        for chunk_id, text, metadata in zip(raw["ids"], raw["documents"], raw["metadatas"], strict=True)
    ]


def get_bm25_retriever() -> BM25Retriever:
    """Rebuilt fresh on every call rather than cached — at this project's
    scale (a handful of PDFs, low thousands of chunks) re-tokenizing the
    active corpus is sub-second, and it guarantees the sparse index can
    never go stale after an ingestion or a version deactivation. Worth
    revisiting with an invalidation-based cache only if the corpus grows
    large enough for rebuild cost to matter."""
    return BM25Retriever.from_documents(
        active_child_documents(), k=settings.retrieval_top_k, preprocess_func=_tokenize
    )


def dense_search(query: str) -> list[Document]:
    """Child-level similarity search (not the auto-expanding
    ParentDocumentRetriever) so the dense side matches BM25's granularity —
    see this module's docstring note in hybrid.py's intro for why that
    matters for fusion. Stamps the same chunk_id field BM25 uses, so
    EnsembleRetriever can recognize when both sides found the same chunk."""
    store = get_chroma_store()
    docs = store.similarity_search(query, k=settings.retrieval_top_k, filter=ACTIVE_FILTER)
    for doc in docs:
        doc.metadata["chunk_id"] = doc.id
    return docs


def get_hybrid_retriever() -> EnsembleRetriever:
    """Weighted RRF fusion of sparse (BM25 — exact terms, policy codes) and
    dense (semantic) search, keyed on the shared chunk_id so genuinely
    overlapping hits from both sides reinforce each other's rank instead of
    being treated as unrelated results."""
    return EnsembleRetriever(
        retrievers=[get_bm25_retriever(), RunnableLambda(dense_search)],
        weights=[settings.bm25_weight, settings.dense_weight],
        id_key="chunk_id",
    )
