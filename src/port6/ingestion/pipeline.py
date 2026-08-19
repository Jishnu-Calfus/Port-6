import shutil
from dataclasses import dataclass
from pathlib import Path

from port6.config import settings
from port6.ingestion import dedup
from port6.ingestion.chunker import split_into_parents
from port6.ingestion.dedup import DedupResult, IngestDecision
from port6.ingestion.loader import load_pdf
from port6.retrieval.vectorstore import get_chroma_store, get_parent_document_retriever


@dataclass
class IngestResult:
    dedup: DedupResult
    num_parent_chunks: int = 0
    num_child_chunks: int = 0


def _document_path(document_id: str, version: int, source_filename: str) -> Path:
    path = Path(settings.documents_dir) / document_id / f"v{version}" / source_filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _version_filter(document_id: str, version: int) -> dict:
    return {"$and": [{"document_id": document_id}, {"version": version}]}


def _deactivate_chroma_version(document_id: str, version: int) -> None:
    """Flips active=False on every child chunk belonging to a superseded
    version, so default (active-only) retrieval stops surfacing it while
    still keeping it around for an explicit historical comparison."""
    collection = get_chroma_store()._collection
    matches = collection.get(where=_version_filter(document_id, version))
    ids = matches["ids"]
    if not ids:
        return
    updated_metadatas = [{**metadata, "active": False} for metadata in matches["metadatas"]]
    collection.update(ids=ids, metadatas=updated_metadatas)


def ingest_document(file_bytes: bytes, source_filename: str) -> IngestResult:
    """The one function the API's upload endpoint calls: dedup check, text
    extraction (with OCR fallback), parent-child chunking, then storage — in
    that order, bailing out early on an exact-duplicate upload."""
    result = dedup.process_upload(file_bytes, source_filename)

    if result.decision == IngestDecision.SKIP_DUPLICATE:
        return IngestResult(dedup=result)

    if result.decision == IngestDecision.NEW_VERSION and result.superseded_version is not None:
        _deactivate_chroma_version(result.meta.document_id, result.superseded_version)

    path = _document_path(result.meta.document_id, result.meta.version, source_filename)
    path.write_bytes(file_bytes)

    pages = load_pdf(path)
    parents = split_into_parents(pages, document_id=result.meta.document_id)
    if not parents:
        return IngestResult(dedup=result)

    for parent in parents:
        parent.metadata["version"] = result.meta.version
        parent.metadata["active"] = True

    get_parent_document_retriever().add_documents(parents)

    child_ids = get_chroma_store()._collection.get(
        where=_version_filter(result.meta.document_id, result.meta.version)
    )["ids"]

    return IngestResult(
        dedup=result,
        num_parent_chunks=len(parents),
        num_child_chunks=len(child_ids),
    )


def delete_document(document_id: str) -> None:
    """Hard delete: removes every version's chunks from Chroma, the
    original files from disk, and the registry entry. Parent docstore blobs
    for the deleted chunks are left as orphaned, unreachable data — harmless
    (nothing in Chroma points to them anymore), and not worth a more complex
    cleanup mechanism at this project's scale."""
    get_chroma_store()._collection.delete(where={"document_id": document_id})

    document_dir = Path(settings.documents_dir) / document_id
    if document_dir.exists():
        shutil.rmtree(document_dir)

    dedup.delete_document(document_id)
