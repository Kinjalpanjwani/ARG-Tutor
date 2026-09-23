from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.api.dependencies import get_container
from app.container import Container
from app.schemas.lesson import TranscriptionResponse

router = APIRouter(prefix="/speech", tags=["speech"])


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...), language: str | None = Form(None), container: Container = Depends(get_container)) -> TranscriptionResponse:
    suffix = Path(file.filename or "audio.webm").suffix or ".webm"
    try:
        with NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            path = Path(temporary.name)
            temporary.write(await file.read())
        detected, text = await container.stt.transcribe(path, language=language)
        return TranscriptionResponse(speech_detected=detected, text=text)
    except Exception as exc:
        raise HTTPException(502, str(exc)) from exc
    finally:
        if "path" in locals():
            path.unlink(missing_ok=True)


@router.post("/synthesize", response_class=FileResponse)
async def synthesize(
    text: str = Form(..., min_length=1), language: str = Form("English"),
    container: Container = Depends(get_container),
) -> FileResponse:
    try:
        path = await container.tts.synthesize(text, language)
        return FileResponse(path, media_type="audio/mpeg", filename="tutor.mp3")
    except Exception as exc:
        raise HTTPException(502, f"Speech synthesis failed: {exc}") from exc
