import re

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from app.api.dependencies import get_container
from app.container import Container
from app.schemas.lesson import FlashcardResponse
from app.teaching import flashcard_pdf

router = APIRouter(prefix="/lessons", tags=["flashcards"])


@router.post("/{session_id}/flashcards", response_model=FlashcardResponse)
async def generate_flashcards(
    session_id: str,
    count: int = Query(default=10, ge=3, le=20),
    container: Container = Depends(get_container),
) -> FlashcardResponse:
    try:
        return await container.flashcards.generate(session_id, count)
    except KeyError as exc:
        raise HTTPException(404, "Lesson session not found") from exc


@router.get("/{session_id}/flashcards/pdf", response_class=Response)
async def download_flashcards_pdf(
    session_id: str,
    count: int = Query(default=10, ge=3, le=20),
    container: Container = Depends(get_container),
) -> Response:
    try:
        lesson = container.flashcards.lesson_store.get(session_id)
        payload = await container.flashcards.generate(session_id, count)
    except KeyError as exc:
        raise HTTPException(404, "Lesson session not found") from exc
    pdf = flashcard_pdf.render(payload.cards, lesson.topic, payload.language or "")
    slug = re.sub(r"[^a-z0-9]+", "-", lesson.topic.lower()).strip("-") or "flashcards"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="flashcards-{slug}.pdf"'},
    )