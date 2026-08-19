"""Playground-only alternate vector backend. Builds an in-memory FAISS
index from the current active chunks, purely to measure recall/latency
against Chroma at this corpus's actual scale — never used in the production
retrieval path (hybrid.py / reranker.py / chain.py)."""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from port6.config import settings
from port6.retrieval.hybrid import active_child_documents
from port6.retrieval.vectorstore import get_embeddings


def build_faiss_index() -> FAISS | None:
    documents = active_child_documents()
    if not documents:
        return None
    return FAISS.from_documents(documents, get_embeddings())


def faiss_search(index: FAISS, query: str) -> list[Document]:
    return index.similarity_search(query, k=settings.retrieval_top_k)
