from pydantic import BaseModel, Field

from app.schemas.common import Source


class DocumentRecord(BaseModel):
    document_id: str
    course_id: str
    document_name: str
    document_type: str
    subject: str | None = None
    student_level: str | None = None
    sha256: str
    chunks_indexed: int
    stored_path: str | None = None
    media_urls: list[str] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    document: DocumentRecord
    duplicate: bool = False


class RetrievalPreviewRequest(BaseModel):
    course_id: str
    query: str = Field(min_length=1, max_length=5000)


class RetrievalPreviewResponse(BaseModel):
    result_count: int
    source_type: str
    sources: list[Source]
