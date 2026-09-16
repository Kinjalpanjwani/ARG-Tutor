from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import get_container
from app.container import Container
from app.schemas.teaching import FollowupResponse, StudentResponseRequest, TeachingResponse, TeachingStartRequest
from app.teaching.engine import InvalidSessionStateError, NonAcademicError
from app.teaching.session import SessionNotFoundError

router = APIRouter(prefix="/teaching", tags=["teaching"])


def teaching_error(exc: Exception) -> HTTPException:
    if isinstance(exc, SessionNotFoundError):
        return HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    if isinstance(exc, NonAcademicError):
        return HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    return HTTPException(status.HTTP_409_CONFLICT, str(exc))


@router.post("/start", response_model=TeachingResponse)
async def start(data: TeachingStartRequest, container: Container = Depends(get_container)) -> TeachingResponse:
    try:
        return await container.teaching.start(data.session_id)
    except (SessionNotFoundError, NonAcademicError, InvalidSessionStateError) as exc:
        raise teaching_error(exc) from exc


@router.post("/{session_id}/next", response_model=TeachingResponse)
async def next_step(session_id: str, container: Container = Depends(get_container)) -> TeachingResponse:
    try:
        return await container.teaching.next(session_id)
    except (SessionNotFoundError, InvalidSessionStateError) as exc:
        raise teaching_error(exc) from exc


@router.post("/{session_id}/respond", response_model=TeachingResponse)
async def respond(session_id: str, data: StudentResponseRequest, container: Container = Depends(get_container)) -> TeachingResponse:
    try:
        return await container.teaching.respond(session_id, data.content, data.is_interruption)
    except (SessionNotFoundError, NonAcademicError, InvalidSessionStateError) as exc:
        raise teaching_error(exc) from exc


@router.post("/{session_id}/followups", response_model=FollowupResponse)
async def followups(session_id: str, count: int = Query(3, ge=1, le=8), container: Container = Depends(get_container)) -> FollowupResponse:
    try:
        return FollowupResponse(questions=await container.followups.generate(session_id, count))
    except SessionNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found") from exc
