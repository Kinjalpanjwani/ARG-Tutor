from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_container
from app.container import Container
from app.schemas.lesson import (
    InterruptionRequest,
    InterruptionResponse,
    LectureStartRequest,
    LectureStartResponse,
    LessonSectionResponse,
    LessonQuizResponse,
    QuizGradeRequest,
    QuizGradeResponse,
)

router = APIRouter(prefix="/lessons", tags=["notebook lesson"])


@router.post("/start", response_model=LectureStartResponse)
async def start_lesson(data: LectureStartRequest, container: Container = Depends(get_container)) -> LectureStartResponse:
    if not container.courses.get(data.course_id):
        raise HTTPException(404, "Course not found")
    try:
        return await container.notebook_lessons.start(data.course_id, data.topic, data.language)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/{session_id}/next-section", response_model=LessonSectionResponse)
async def next_lesson_section(session_id: str, container: Container = Depends(get_container)) -> LessonSectionResponse:
    try:
        return await container.notebook_lessons.next_section(session_id)
    except KeyError as exc:
        raise HTTPException(404, "Lesson session not found") from exc


@router.post("/{session_id}/interrupt", response_model=InterruptionResponse)
async def interrupt_lesson(
    session_id: str, data: InterruptionRequest,
    container: Container = Depends(get_container),
) -> InterruptionResponse:
    try:
        return await container.notebook_lessons.interrupt(
            session_id, data.question, data.current_chunk_index
        )
    except KeyError as exc:
        raise HTTPException(404, "Lesson session not found") from exc


@router.post("/{session_id}/quiz", response_model=LessonQuizResponse)
async def generate_lesson_quiz(session_id: str, count: int = 5, container: Container = Depends(get_container)) -> LessonQuizResponse:
    if count not in {3, 5, 10}:
        raise HTTPException(422, "Quiz length must be 3, 5, or 10")
    try:
        return await container.notebook_lessons.quiz(session_id, count)
    except KeyError as exc:
        raise HTTPException(404, "Lesson session not found") from exc


@router.post("/{session_id}/quiz/grade", response_model=QuizGradeResponse)
async def grade_lesson_quiz(session_id: str, data: QuizGradeRequest, container: Container = Depends(get_container)) -> QuizGradeResponse:
    try:
        return await container.notebook_lessons.grade_quiz(session_id, data.question, data.answer)
    except KeyError as exc:
        raise HTTPException(404, "Lesson session not found") from exc
