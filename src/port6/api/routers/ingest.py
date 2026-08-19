from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from port6.config import settings
from port6.ingestion import dedup, pipeline
from port6.schemas import DocumentMeta

router = APIRouter(prefix="/documents", tags=["documents"])


class IngestResponse(BaseModel):
    decision: str
    document: DocumentMeta
    superseded_version: int | None
    num_parent_chunks: int
    num_child_chunks: int


@router.post("", response_model=IngestResponse, status_code=201)
async def upload_document(file: UploadFile) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()
    result = pipeline.ingest_document(file_bytes, file.filename)
    return IngestResponse(
        decision=result.dedup.decision.value,
        document=result.dedup.meta,
        superseded_version=result.dedup.superseded_version,
        num_parent_chunks=result.num_parent_chunks,
        num_child_chunks=result.num_child_chunks,
    )


@router.get("", response_model=list[DocumentMeta])
async def list_documents() -> list[DocumentMeta]:
    return dedup.list_documents()


@router.get("/{document_id}/versions/{version}/file")
async def get_document_file(document_id: str, version: int) -> FileResponse:
    try:
        meta = dedup.get_meta(document_id, version)
    except KeyError:
        raise HTTPException(status_code=404, detail="Document version not found") from None

    file_path = Path(settings.documents_dir) / document_id / f"v{version}" / meta.source_filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Original file not found on disk")

    return FileResponse(file_path, media_type="application/pdf", filename=meta.source_filename)


@router.post("/{document_id}/versions/{version}/activate", response_model=DocumentMeta)
async def activate_version(document_id: str, version: int) -> DocumentMeta:
    try:
        return dedup.reactivate_version(document_id, version)
    except KeyError:
        raise HTTPException(status_code=404, detail="Document version not found") from None


@router.post("/{document_id}/deactivate", response_model=DocumentMeta)
async def deactivate_document(document_id: str) -> DocumentMeta:
    meta = dedup.deactivate(document_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Document has no active version")
    return meta


@router.delete("/{document_id}", status_code=204)
async def delete_document(document_id: str) -> None:
    pipeline.delete_document(document_id)
