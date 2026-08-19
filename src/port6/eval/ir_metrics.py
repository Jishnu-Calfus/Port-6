"""Deterministic (no LLM call) retrieval and citation metrics — cheap enough
to run across many pipeline-config sweeps (chunking strategy, HNSW params,
FAISS vs Chroma) without racking up API cost, unlike the LLM-judged metrics
in ragas_eval.py."""

from langchain_core.documents import Document

from port6.schemas import Citation


def hit_at_k(retrieved_docs: list[Document], expected_document_id: str) -> bool:
    """Did the expected document appear anywhere in the retrieved set?
    (Whatever 'k' the caller already sliced to — usually the retriever's own
    top-k/top-n, not re-sliced here.)"""
    return any(doc.metadata.get("document_id") == expected_document_id for doc in retrieved_docs)


def reciprocal_rank(retrieved_docs: list[Document], expected_document_id: str) -> float:
    """1/rank of the first chunk from the expected document, 0.0 if it never
    appears. Averaging this over a query set gives MRR."""
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc.metadata.get("document_id") == expected_document_id:
            return 1.0 / rank
    return 0.0


def citation_accuracy(citations: list[Citation], expected_document_id: str, expected_page: int | None) -> bool:
    """Does the chain's final citation list — what the user actually sees —
    include the expected document (and page, if specified)? This is the
    deterministic stand-in for what a mentor manually checks by eye."""
    for citation in citations:
        if citation.document_id != expected_document_id:
            continue
        if expected_page is None or citation.page_number == expected_page:
            return True
    return False
