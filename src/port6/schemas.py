from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class IntentLabel(StrEnum):
    OFF_TOPIC = "off_topic"
    JAILBREAK = "jailbreak"
    SENSITIVE = "sensitive"
    DIALOG_INTENT = "dialog_intent"
    IT_QUESTION = "it_question"


class DialogSubtype(StrEnum):
    GREETING = "greeting"
    HELP = "help"
    BYE = "bye"


class DocumentMeta(BaseModel):
    document_id: str
    version: int
    active: bool
    source_filename: str
    content_hash: str
    ingested_at: datetime


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    page_number: int
    parent_id: str | None = None


class Citation(BaseModel):
    document_id: str
    source_filename: str
    page_number: int
    snippet: str


class IntentResult(BaseModel):
    label: IntentLabel
    dialog_subtype: DialogSubtype | None = None


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    refused: bool = False
