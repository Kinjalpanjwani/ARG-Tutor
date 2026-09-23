from fastapi import APIRouter, Depends, HTTPException
from app.api.dependencies import get_container
from app.container import Container
from app.schemas.vision import VisionStateUpdate, VisionStatusResponse
from app.teaching.session import SessionNotFoundError

router = APIRouter(prefix="/vision", tags=["vision"])

@router.post("/{session_id}", response_model=VisionStatusResponse)
async def update_vision_state(
    session_id: str,
    update: VisionStateUpdate,
    container: Container = Depends(get_container),
) -> VisionStatusResponse:
    """Receive a vision update (attention & mood) for a teaching session.

    The update is stored in the SessionStore and the current vision‑related
    status is returned so the frontend can react (e.g., display a pause).
    """
    try:
        session = container.sessions.update_vision_state(session_id, update)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    return VisionStatusResponse(
        session_id=session_id,
        attention=session.latest_attention,
        mood=session.latest_mood,
        paused=session.paused_for_no_face,
        cooldown=session.checkin_cooldown,
        awaiting_checkin_reply=session.awaiting_checkin_reply,
    )

