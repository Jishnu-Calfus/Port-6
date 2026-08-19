"""RAGAS wrapper for generation-quality metrics — faithfulness,
answer_relevancy, context_precision, context_recall. LLM-judged, so this is
comparatively expensive; used for judging final answer quality on the
golden set, not for sweeping many retrieval-only configs (see ir_metrics.py
for the cheap, deterministic version of that).

ragas==0.4.3 unconditionally imports langchain_community.chat_models.vertexai
at module load time. That module no longer exists in current
langchain-community (ChatVertexAI moved to the standalone
langchain-google-vertexai package) — a known, currently open upstream ragas
bug (github.com/vibrantlabsai/ragas/issues/2745), not something fixable from
our side. We don't use VertexAI at all, so this stub satisfies the dead
import without downgrading langchain-community (which would conflict with
our langchain 1.x / langchain-classic stack)."""

import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class _UnusedChatVertexAI:  # pragma: no cover - never instantiated, just satisfies the import
        pass

    _vertexai_stub.ChatVertexAI = _UnusedChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

from functools import lru_cache

from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)

from port6.config import settings


@lru_cache
def _get_ragas_llm():
    # AsyncOpenAI, not OpenAI — score()/ascore() run the async path
    # internally (agenerate()), which raises against a sync client.
    # timeout is explicit because the SDK's own default is 600s *per
    # request* with retries on top — a single network hiccup during a
    # multi-question eval run can otherwise hang for tens of minutes
    # instead of failing fast into run_eval()'s try/except.
    client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=30, max_retries=1)
    return llm_factory(settings.openai_chat_model, client=client)


@lru_cache
def _get_ragas_embeddings():
    client = AsyncOpenAI(api_key=settings.openai_api_key, timeout=30, max_retries=1)
    return RagasOpenAIEmbeddings(client=client, model=settings.openai_embedding_model)


def evaluate_generation(question: str, answer: str, contexts: list[str], reference_answer: str) -> dict[str, float]:
    """The 4 generation-quality metrics for one (question, answer, retrieved
    contexts, reference answer) tuple. reference_answer is the golden set's
    known-correct answer — used by context_precision/context_recall to judge
    whether retrieval found what was actually needed, independent of
    whatever the LLM ultimately wrote."""
    llm = _get_ragas_llm()

    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=_get_ragas_embeddings())
    context_precision = ContextPrecisionWithReference(llm=llm)
    context_recall = ContextRecall(llm=llm)

    return {
        "faithfulness": faithfulness.score(
            user_input=question, response=answer, retrieved_contexts=contexts
        ).value,
        "answer_relevancy": answer_relevancy.score(user_input=question, response=answer).value,
        "context_precision": context_precision.score(
            user_input=question, reference=reference_answer, retrieved_contexts=contexts
        ).value,
        "context_recall": context_recall.score(
            user_input=question, retrieved_contexts=contexts, reference=reference_answer
        ).value,
    }
