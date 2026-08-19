import requests

from port6.config import settings
from port6.schemas import AnswerResponse, DocumentMeta


def query(message: str) -> AnswerResponse:
    response = requests.post(f"{settings.api_base_url}/query", json={"message": message}, timeout=60)
    response.raise_for_status()
    return AnswerResponse.model_validate(response.json())


def list_documents() -> list[DocumentMeta]:
    response = requests.get(f"{settings.api_base_url}/documents", timeout=30)
    response.raise_for_status()
    return [DocumentMeta.model_validate(item) for item in response.json()]


def upload_document(file_bytes: bytes, filename: str) -> dict:
    response = requests.post(
        f"{settings.api_base_url}/documents",
        files={"file": (filename, file_bytes, "application/pdf")},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def get_document_file(document_id: str, version: int) -> bytes:
    response = requests.get(f"{settings.api_base_url}/documents/{document_id}/versions/{version}/file", timeout=30)
    response.raise_for_status()
    return response.content


def activate_version(document_id: str, version: int) -> DocumentMeta:
    response = requests.post(
        f"{settings.api_base_url}/documents/{document_id}/versions/{version}/activate", timeout=30
    )
    response.raise_for_status()
    return DocumentMeta.model_validate(response.json())


def deactivate_document(document_id: str) -> DocumentMeta:
    response = requests.post(f"{settings.api_base_url}/documents/{document_id}/deactivate", timeout=30)
    response.raise_for_status()
    return DocumentMeta.model_validate(response.json())


def delete_document(document_id: str) -> None:
    response = requests.delete(f"{settings.api_base_url}/documents/{document_id}", timeout=30)
    response.raise_for_status()
