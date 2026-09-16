import hashlib
import re
from pathlib import Path

from fastapi import UploadFile

from app.core.constants import ALLOWED_EXTENSIONS


class InvalidDocumentError(ValueError):
    pass


def safe_filename(name: str) -> str:
    name = Path(name).name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "document"


async def read_validated_upload(upload: UploadFile, max_bytes: int) -> tuple[str, bytes]:
    filename = safe_filename(upload.filename or "document")
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise InvalidDocumentError("Supported files: PDF, PNG, JPG, JPEG, PPTX, TXT, and Markdown")
    content = await upload.read(max_bytes + 1)
    if not content:
        raise InvalidDocumentError("Uploaded file is empty")
    if len(content) > max_bytes:
        raise InvalidDocumentError("Uploaded file exceeds the configured size limit")
    return filename, content


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
