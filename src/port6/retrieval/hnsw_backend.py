"""Playground-only HNSW parameter sweep. Builds throwaway, in-memory Chroma
collections (never persisted, never touching the production collection)
with different (M / ef_construction / ef_search) settings, populated from
the same active corpus, to measure the real recall-vs-latency tradeoff each
knob controls at this corpus's actual scale.

M and ef_construction are baked into the HNSW graph at build time, so
comparing them requires a full rebuild per configuration — ef_search alone
could be changed on a live collection via .modify(), but building fresh
collections here keeps every configuration fully independent and lets all
three knobs be swept the same way. Re-embedding hits the same Redis
embedding cache as production, so sweeping costs no extra OpenAI calls for
text already embedded once."""

from dataclasses import dataclass

import chromadb
from langchain_chroma import Chroma

from port6.retrieval.hybrid import active_child_documents
from port6.retrieval.vectorstore import get_embeddings


@dataclass
class HnswConfig:
    label: str
    m: int
    ef_construction: int
    ef_search: int
    tradeoff: str


DEFAULT_SWEEP = [
    HnswConfig(
        "Low",
        m=4,
        ef_construction=20,
        ef_search=10,
        tradeoff="Small graph, few candidates checked per search — fastest and cheapest to build, but most likely to miss the right chunk on a harder or larger corpus.",
    ),
    HnswConfig(
        "Chroma default",
        m=16,
        ef_construction=100,
        ef_search=10,
        tradeoff="Chroma's out-of-the-box setting — a balance point with no tuning applied.",
    ),
    HnswConfig(
        "High",
        m=48,
        ef_construction=200,
        ef_search=100,
        tradeoff="Large, densely-connected graph and many candidates checked per search — highest recall, at real memory and per-query latency cost.",
    ),
]


def build_hnsw_variant(config: HnswConfig) -> Chroma | None:
    documents = active_child_documents()
    if not documents:
        return None

    client = chromadb.EphemeralClient()
    store = Chroma(
        collection_name=f"hnsw_sweep_{config.m}_{config.ef_construction}_{config.ef_search}",
        embedding_function=get_embeddings(),
        client=client,
        collection_configuration={
            "hnsw": {
                "space": "cosine",
                "ef_construction": config.ef_construction,
                "max_neighbors": config.m,
                "ef_search": config.ef_search,
            }
        },
    )
    store.add_documents(documents)
    return store
