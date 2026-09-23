from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_container
from app.container import Container
from app.schemas.sessions import SessionCreate, TeachingSession
from app.schemas.vision import VisionStateUpdate
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


@router.post("/{session_id}/vision-state")
async def update_vision_state(
    session_id: str,
    data: VisionStateUpdate,
    container: Container = Depends(get_container),
):
    session = None
    try:
        session = container.sessions.update_vision_state(session_id, data)
        return {
            "session_id": session_id,
            "attention": session.latest_attention,
            "mood": session.latest_mood,
            "paused": session.paused_for_no_face,
            "cooldown": session.checkin_cooldown,
            "awaiting_checkin_reply": session.awaiting_checkin_reply,
        }
    except SessionNotFoundError:
        pass

    try:
        lesson = container.notebook_lessons.store.update_vision_state(session_id, data)
        checkin_question = await container.notebook_lessons.maybe_fire_checkin(session_id)
        return {
            "session_id": session_id,
            "attention": lesson.latest_attention,
            "mood": lesson.latest_mood,
            "paused": lesson.paused_for_no_face,
            "cooldown": lesson.checkin_cooldown,
            "awaiting_checkin_reply": lesson.awaiting_checkin_reply,
            "checkin_question": checkin_question,
            "language": container.notebook_lessons.response_language(lesson),
        }
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")

