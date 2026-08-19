"""Playground-only comparisons — retrieval strategy (naive dense vs hybrid
vs hybrid+rerank), vector backend (Chroma vs FAISS), and HNSW parameters
(M / ef_construction / ef_search) — all measured with the same
deterministic IR metrics (ir_metrics.py) used for the production eval,
against the real golden set and real corpus.

These deliberately don't touch generation: RAGAS's LLM-judged metrics are
about answer quality, which isn't what changes between these configs — IR
metrics (free, deterministic, no LLM calls) are the right tool for isolating
a pure retrieval change."""

import time
from dataclasses import dataclass

from port6.config import settings
from port6.eval.ir_metrics import hit_at_k, reciprocal_rank
from port6.eval.runner import load_golden_set
from port6.retrieval.faiss_backend import build_faiss_index, faiss_search
from port6.retrieval.hnsw_backend import DEFAULT_SWEEP, build_hnsw_variant
from port6.retrieval.hybrid import dense_search, get_hybrid_retriever
from port6.retrieval.reranker import get_reranking_retriever
from port6.retrieval.vectorstore import get_chroma_store


@dataclass
class StrategyResult:
    name: str
    recall_at_k: float
    mrr: float
    avg_latency_seconds: float


@dataclass
class HnswResult(StrategyResult):
    m: int
    ef_construction: int
    ef_search: int
    tradeoff: str


def _it_questions(golden_set: list[dict] | None) -> list[dict]:
    golden_set = golden_set if golden_set is not None else load_golden_set()
    return [item for item in golden_set if not item["expect_refusal"]]


def _compute_metrics(search_fn, questions: list[dict]) -> dict:
    hits, rrs, latencies = [], [], []
    for item in questions:
        start = time.monotonic()
        docs = search_fn(item["question"])
        latencies.append(time.monotonic() - start)
        hits.append(hit_at_k(docs, item["expected_document_id"]))
        rrs.append(reciprocal_rank(docs, item["expected_document_id"]))

    n = len(questions) or 1
    return {
        "recall_at_k": sum(hits) / n,
        "mrr": sum(rrs) / n,
        "avg_latency_seconds": sum(latencies) / n,
    }


def _score_strategy(name: str, search_fn, questions: list[dict]) -> StrategyResult:
    return StrategyResult(name=name, **_compute_metrics(search_fn, questions))


def compare_retrieval_strategies(golden_set: list[dict] | None = None) -> list[StrategyResult]:
    questions = _it_questions(golden_set)
    strategies = {
        "Naive (dense only)": dense_search,
        "Hybrid (BM25 + dense)": lambda q: get_hybrid_retriever().invoke(q),
        "Hybrid + Rerank (production)": lambda q: get_reranking_retriever().invoke(q),
    }
    return [_score_strategy(name, fn, questions) for name, fn in strategies.items()]


def compare_vector_backends(golden_set: list[dict] | None = None) -> list[StrategyResult]:
    questions = _it_questions(golden_set)
    faiss_index = build_faiss_index()

    backends = {
        "Chroma (production)": lambda q: get_chroma_store().similarity_search(
            q, k=settings.retrieval_top_k, filter={"active": True}
        ),
    }
    if faiss_index is not None:
        backends["FAISS (playground)"] = lambda q: faiss_search(faiss_index, q)

    return [_score_strategy(name, fn, questions) for name, fn in backends.items()]


def compare_hnsw_configs(golden_set: list[dict] | None = None) -> list[HnswResult]:
    questions = _it_questions(golden_set)
    results = []
    for config in DEFAULT_SWEEP:
        store = build_hnsw_variant(config)
        if store is None:
            continue
        search_fn = lambda q, s=store: s.similarity_search(q, k=settings.retrieval_top_k)
        metrics = _compute_metrics(search_fn, questions)
        results.append(
            HnswResult(
                name=config.label,
                m=config.m,
                ef_construction=config.ef_construction,
                ef_search=config.ef_search,
                tradeoff=config.tradeoff,
                **metrics,
            )
        )
    return results
