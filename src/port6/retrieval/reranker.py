from functools import lru_cache

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.retrievers import BaseRetriever

from port6.config import settings
from port6.retrieval.hybrid import get_hybrid_retriever


@lru_cache
def get_cross_encoder() -> HuggingFaceCrossEncoder:
    """Loaded once and cached — this loads real model weights, not
    something to repeat per query."""
    return HuggingFaceCrossEncoder(model_name=settings.reranker_model)


def get_reranking_retriever() -> BaseRetriever:
    """Wraps the fused hybrid retriever with a cross-encoder that scores
    query-chunk relevance jointly, rather than combining two independent
    rankers' opinions the way RRF fusion does — this is what corrects cases
    where hybrid fusion's blended ranking gets the top result wrong."""
    reranker = CrossEncoderReranker(model=get_cross_encoder(), top_n=settings.rerank_top_n)
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=get_hybrid_retriever(),
    )
