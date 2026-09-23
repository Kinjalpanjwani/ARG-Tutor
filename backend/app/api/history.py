from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_container
from app.container import Container
from app.schemas.lesson import (
    LessonArchivePayload,
    LessonHistoryEntry,
    LessonRecord,
    QuizResultsPayload,
)

router = APIRouter(prefix="/lessons", tags=["lesson history"])


@router.get("/history", response_model=list[LessonHistoryEntry])
def list_lesson_history(
    course_id: str | None = Query(None),
    container: Container = Depends(get_container),
) -> list[LessonHistoryEntry]:
    if container.history is None:
        raise HTTPException(503, "History service unavailable")
    return container.history.list(course_id)


@router.get("/history/{session_id}", response_model=LessonRecord)
def get_lesson_history(
    session_id: str,
    container: Container = Depends(get_container),
) -> LessonRecord:
    try:
        return container.history.get(session_id)
    except KeyError as exc:
        raise HTTPException(404, "Lesson archive not found") from exc


@router.post("/{session_id}/archive", response_model=LessonRecord)
def archive_lesson(
    session_id: str,
    data: LessonArchivePayload,
    container: Container = Depends(get_container),
) -> LessonRecord:
    try:
        return container.history.archive(session_id, data.blocks)
    except KeyError as exc:
        raise HTTPException(404, "Lesson session not found") from exc


@router.post("/{session_id}/quiz-result", response_model=LessonRecord)
def save_quiz_result(
    session_id: str,
    data: QuizResultsPayload,
    container: Container = Depends(get_container),
) -> LessonRecord:
    try:
        return container.history.update_quiz(session_id, data)
    except KeyError as exc:
        raise HTTPException(404, "Lesson session not found") from exc