from functools import lru_cache

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from port6.config import settings
from port6.guardrails.intent import classify_intent
from port6.ingestion.dedup import get_meta
from port6.rag.prompts import DIALOG_REPLIES, GENERATION_SYSTEM_PROMPT, NO_CONTEXT_MESSAGE, REFUSAL_MESSAGES
from port6.retrieval.reranker import get_reranking_retriever
from port6.retrieval.vectorstore import get_parent_document_retriever
from port6.schemas import AnswerResponse, Citation, IntentLabel

_CONTEXT_SEPARATOR = "\n\n---\n\n"


class _GenerationResult(BaseModel):
    answer: str
    answered: bool
    """False when falling back to the "I don't have information..." case —
    lets the chain suppress citations on a non-answer instead of guessing
    from the prose whether the model actually refused."""


@lru_cache
def _get_generation_llm():
    llm = ChatOpenAI(model=settings.openai_chat_model, api_key=settings.openai_api_key, temperature=0)
    return llm.with_structured_output(_GenerationResult)


def _dedup_parent_ids(child_docs: list[Document]) -> list[str]:
    """Multiple reranked children often belong to the same parent — collapse
    them to one entry each, preserving rank order, so that parent isn't fed
    into the context (or cited) more than once."""
    seen: list[str] = []
    for doc in child_docs:
        parent_id = doc.metadata.get("doc_id")
        if parent_id and parent_id not in seen:
            seen.append(parent_id)
    return seen


def _representative_child_by_parent(child_docs: list[Document]) -> dict[str, Document]:
    """The highest-ranked child chunk for each parent — used as the
    citation's snippet, since it's the specific fragment that actually
    matched, not the whole (larger) parent context."""
    representative: dict[str, Document] = {}
    for doc in child_docs:
        parent_id = doc.metadata.get("doc_id")
        if parent_id and parent_id not in representative:
            representative[parent_id] = doc
    return representative


def _build_citation(child_doc: Document, parent_doc: Document) -> Citation:
    document_id = child_doc.metadata["document_id"]
    version = child_doc.metadata["version"]
    meta = get_meta(document_id, version)
    return Citation(
        document_id=document_id,
        source_filename=meta.source_filename,
        page_number=parent_doc.metadata.get("page_number", child_doc.metadata.get("page_number", 0)),
        snippet=child_doc.page_content,
    )


def answer_query(message: str) -> AnswerResponse:
    """The one function the API's /query endpoint calls. Runs the intent
    check first; only a genuine it_question reaches retrieval, reranking,
    parent expansion, and generation — everything else gets a fixed
    scripted response with no retrieval or generation cost at all."""
    intent = classify_intent(message)

    if intent.label in REFUSAL_MESSAGES:
        return AnswerResponse(answer=REFUSAL_MESSAGES[intent.label], citations=[], refused=True)

    if intent.label == IntentLabel.DIALOG_INTENT:
        return AnswerResponse(answer=DIALOG_REPLIES[intent.dialog_subtype], citations=[], refused=False)

    child_docs = get_reranking_retriever().invoke(message)
    if not child_docs:
        return AnswerResponse(answer=NO_CONTEXT_MESSAGE, citations=[], refused=True)

    parent_ids = _dedup_parent_ids(child_docs)
    parent_docs = get_parent_document_retriever().docstore.mget(parent_ids)
    representative_child = _representative_child_by_parent(child_docs)

    context_parts: list[str] = []
    citations: list[Citation] = []
    for parent_id, parent_doc in zip(parent_ids, parent_docs, strict=True):
        if parent_doc is None:
            continue
        context_parts.append(parent_doc.page_content)
        citations.append(_build_citation(representative_child[parent_id], parent_doc))

    if not context_parts:
        return AnswerResponse(answer=NO_CONTEXT_MESSAGE, citations=[], refused=True)

    system_prompt = GENERATION_SYSTEM_PROMPT.format(context=_CONTEXT_SEPARATOR.join(context_parts))
    result = _get_generation_llm().invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]
    )

    if not result.answered:
        return AnswerResponse(answer=result.answer, citations=[], refused=True)

    return AnswerResponse(answer=result.answer, citations=citations, refused=False)
