from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_container
from app.container import Container
from app.schemas.sessions import SessionCreate, TeachingSession
from app.teaching.session import SessionNotFoundError

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=TeachingSession, status_code=status.HTTP_201_CREATED)
async def create_session(data: SessionCreate, container: Container = Depends(get_container)) -> TeachingSession:
    if data.course_id and not container.courses.get(data.course_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Course not found")
    return container.sessions.create(data)


@router.get("/{session_id}", response_model=TeachingSession)
async def get_session(session_id: str, container: Container = Depends(get_container)) -> TeachingSession:
    try:
        return container.sessions.get(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from exc
