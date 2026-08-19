import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from port6.caching.redis_client import get_redis_client
from port6.schemas import DocumentMeta

_LOCK_TIMEOUT = 30
_LOCK_BLOCKING_TIMEOUT = 10

_KEY_HASH_INDEX = "port6:hash_index"
_KEY_DOCUMENT_IDS = "port6:documents"


def _versions_key(document_id: str) -> str:
    return f"port6:doc:{document_id}:versions"


def _active_version_key(document_id: str) -> str:
    return f"port6:doc:{document_id}:active_version"


class IngestDecision(StrEnum):
    SKIP_DUPLICATE = "skip_duplicate"
    NEW_VERSION = "new_version"
    NEW_DOCUMENT = "new_document"


@dataclass
class DedupResult:
    decision: IngestDecision
    meta: DocumentMeta
    superseded_version: int | None = None


def compute_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def normalize_document_id(source_filename: str) -> str:
    """Same-named re-uploads are treated as a new version of the same
    logical document (e.g. an HR policy PDF getting updated) rather than a
    brand new one — this is the stable key that lineage hangs off of."""
    stem = Path(source_filename).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return slug or "document"


def _load_meta(document_id: str, version: int) -> DocumentMeta:
    raw = get_redis_client().hget(_versions_key(document_id), str(version))
    if raw is None:
        raise KeyError(f"No stored metadata for {document_id} v{version}")
    return DocumentMeta.model_validate_json(raw)


def _save_meta(meta: DocumentMeta) -> None:
    r = get_redis_client()
    r.hset(_versions_key(meta.document_id), str(meta.version), meta.model_dump_json())
    r.sadd(_KEY_DOCUMENT_IDS, meta.document_id)


def process_upload(file_bytes: bytes, source_filename: str) -> DedupResult:
    """The one entry point ingestion goes through: decides whether this
    upload is an exact duplicate (skip), a new version of an existing
    document (deactivate the old one), or a brand new document. Wrapped in a
    per-document lock so two concurrent uploads of the same file can't both
    slip past the duplicate check before either commits."""
    content_hash = compute_hash(file_bytes)
    document_id = normalize_document_id(source_filename)
    r = get_redis_client()

    with r.lock(f"port6:lock:doc:{document_id}", timeout=_LOCK_TIMEOUT, blocking_timeout=_LOCK_BLOCKING_TIMEOUT):
        existing = r.hget(_KEY_HASH_INDEX, content_hash)
        if existing is not None:
            existing_document_id, existing_version = existing.decode().split(":")
            meta = _load_meta(existing_document_id, int(existing_version))
            return DedupResult(decision=IngestDecision.SKIP_DUPLICATE, meta=meta)

        active_version_raw = r.get(_active_version_key(document_id))
        previous_active_version = int(active_version_raw) if active_version_raw else None
        new_version = (previous_active_version or 0) + 1

        meta = DocumentMeta(
            document_id=document_id,
            version=new_version,
            active=True,
            source_filename=source_filename,
            content_hash=content_hash,
            ingested_at=datetime.now(UTC),
        )

        if previous_active_version is not None:
            old_meta = _load_meta(document_id, previous_active_version)
            old_meta.active = False
            _save_meta(old_meta)

        _save_meta(meta)
        r.set(_active_version_key(document_id), new_version)
        r.hset(_KEY_HASH_INDEX, content_hash, f"{document_id}:{new_version}")

        decision = IngestDecision.NEW_VERSION if previous_active_version else IngestDecision.NEW_DOCUMENT
        return DedupResult(decision=decision, meta=meta, superseded_version=previous_active_version)


def get_meta(document_id: str, version: int) -> DocumentMeta:
    """Public lookup the RAG chain uses to resolve a retrieved chunk's
    document_id/version back to its human-readable source_filename for
    citations."""
    return _load_meta(document_id, version)


def list_documents() -> list[DocumentMeta]:
    """Every version of every known document — the HR view groups these by
    document_id to show version history and which one is active."""
    r = get_redis_client()
    document_ids = [d.decode() for d in r.smembers(_KEY_DOCUMENT_IDS)]
    metas: list[DocumentMeta] = []
    for document_id in document_ids:
        raw_versions = r.hgetall(_versions_key(document_id))
        metas.extend(DocumentMeta.model_validate_json(raw) for raw in raw_versions.values())
    return sorted(metas, key=lambda m: (m.document_id, m.version))


def reactivate_version(document_id: str, version: int) -> DocumentMeta:
    """Rolls the active version back/forward to `version`, deactivating
    whatever was previously active. Used to undo an accidental update or
    intentionally revert to a prior policy version."""
    r = get_redis_client()
    current_active_raw = r.get(_active_version_key(document_id))
    if current_active_raw is not None and int(current_active_raw) != version:
        old_meta = _load_meta(document_id, int(current_active_raw))
        old_meta.active = False
        _save_meta(old_meta)

    meta = _load_meta(document_id, version)
    meta.active = True
    _save_meta(meta)
    r.set(_active_version_key(document_id), version)
    return meta


def deactivate(document_id: str) -> DocumentMeta | None:
    """Retires a document family entirely — no version stays active. Used
    for a manual HR 'remove this policy' action, distinct from deleting it
    outright."""
    r = get_redis_client()
    active_raw = r.get(_active_version_key(document_id))
    if active_raw is None:
        return None
    meta = _load_meta(document_id, int(active_raw))
    meta.active = False
    _save_meta(meta)
    r.delete(_active_version_key(document_id))
    return meta


def delete_document(document_id: str) -> None:
    """Hard delete of the whole document family from the registry. The
    caller (the ingest router) is also responsible for deleting the
    matching chunks from Chroma — this only clears the bookkeeping side."""
    r = get_redis_client()
    versions_raw = r.hgetall(_versions_key(document_id))
    for raw in versions_raw.values():
        meta = DocumentMeta.model_validate_json(raw)
        r.hdel(_KEY_HASH_INDEX, meta.content_hash)
    r.delete(_versions_key(document_id))
    r.delete(_active_version_key(document_id))
    r.srem(_KEY_DOCUMENT_IDS, document_id)
