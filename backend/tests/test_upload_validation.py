import io

import pytest
from fastapi import UploadFile

from app.utils.files import InvalidDocumentError, read_validated_upload


@pytest.mark.asyncio
async def test_invalid_file_upload_is_rejected() -> None:
    upload = UploadFile(filename="malware.exe", file=io.BytesIO(b"not allowed"))
    with pytest.raises(InvalidDocumentError):
        await read_validated_upload(upload, 1024)
