from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_container
from app.container import Container
from app.schemas.teaching import QuizResponse
from app.teaching.session import SessionNotFoundError

router = APIRouter(prefix="/quiz", tags=["quiz"])


@router.post("/generate", response_model=QuizResponse)
async def generate_quiz(
    session_id: str, count: int = Query(5, ge=1, le=10),
    container: Container = Depends(get_container),
) -> QuizResponse:
    try:
        return await container.quizzes.generate(session_id, count)
    except SessionNotFoundError as exc:
        raise HTTPException(404, "Session not found") from exc
