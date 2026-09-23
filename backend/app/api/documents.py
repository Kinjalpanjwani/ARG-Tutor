from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.dependencies import get_container
from app.container import Container
from app.rag.loader import DocumentLoadError
from app.schemas.documents import (
    DocumentUploadResponse,
    RetrievalPreviewRequest,
    RetrievalPreviewResponse,
)
from app.utils.files import InvalidDocumentError, read_validated_upload, safe_filename

router = APIRouter(prefix="/documents", tags=["documents"])


@router.delete("/{document_id}")
async def remove_document(
    document_id: str, course_id: str,
    container: Container = Depends(get_container),
) -> dict[str, int | str]:
    try:
        record, removed = container.ingestion.remove(document_id, course_id)
        return {"document_id": record.document_id, "chunks_removed": removed}
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Material not found") from exc


@router.post("/retrieve-preview", response_model=RetrievalPreviewResponse)
async def retrieval_preview(
    data: RetrievalPreviewRequest,
    container: Container = Depends(get_container),
) -> RetrievalPreviewResponse:
    if not container.courses.get(data.course_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Course not found")
    result = container.retriever.retrieve(data.query, data.course_id)
    return RetrievalPreviewResponse(
        result_count=len(result.sources),
        source_type=result.source_type,
        sources=result.sources,
    )


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    course_id: str = Form(...), document_type: str = Form("course_document"),
    subject: str | None = Form(None), student_level: str | None = Form(None),
    file: UploadFile = File(...), container: Container = Depends(get_container),
) -> DocumentUploadResponse:
    if not container.courses.get(course_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Course not found")
    try:
        filename, content = await read_validated_upload(file, container.settings.max_upload_mb * 1024 * 1024)
        destination = container.runtime_dir / "uploads" / f"{uuid4()}_{safe_filename(filename)}"
        destination.write_bytes(content)
        record, duplicate = container.ingestion.ingest(
            destination, content, course_id, document_type, subject, student_level, filename
        )
        if duplicate:
            destination.unlink(missing_ok=True)
        return DocumentUploadResponse(document=record, duplicate=duplicate)
    except (InvalidDocumentError, DocumentLoadError, ValueError) as exc:
        if "destination" in locals():
            destination.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
