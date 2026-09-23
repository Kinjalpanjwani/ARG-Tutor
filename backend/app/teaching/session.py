from datetime import datetime, timezone
import logging
from threading import RLock

logger = logging.getLogger(__name__)

from app.core.constants import (
    AttentionState,
    MoodState,
    SessionStatus,
    VISION_NO_FACE_PAUSE_STREAK,
)

from app.schemas.sessions import SessionCreate, TeachingSession
from app.schemas.vision import VisionStateUpdate


class SessionNotFoundError(KeyError):
    pass


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, TeachingSession] = {}
        self._lock = RLock()

    def create(self, request: SessionCreate) -> TeachingSession:
        session = TeachingSession(**request.model_dump())
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> TeachingSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if not session:
            raise SessionNotFoundError(session_id)
        return session

    def save(self, session: TeachingSession) -> TeachingSession:
        session.updated_at = datetime.now(timezone.utc)
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def update_vision_state(self, session_id: str, update: VisionStateUpdate) -> TeachingSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise SessionNotFoundError(session_id)

            session.latest_attention = update.attention
            session.latest_mood = update.mood
            session.vision_history.append({
                "attention": str(update.attention),
                "mood": str(update.mood),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            if len(session.vision_history) > 20:
                session.vision_history = session.vision_history[-20:]

            if update.attention == AttentionState.NO_FACE:
                session.no_face_streak += 1
                if session.no_face_streak >= VISION_NO_FACE_PAUSE_STREAK:
                    if session.status not in {SessionStatus.PAUSED, SessionStatus.COMPLETED}:
                        session.status = SessionStatus.PAUSED
                        session.paused_for_no_face = True
            else:
                session.no_face_streak = 0
                if session.paused_for_no_face:
                    session.paused_for_no_face = False
                    session.resumed_from_no_face = True
                    session.status = SessionStatus.TEACHING

            session.updated_at = datetime.now(timezone.utc)
            self._sessions[session.session_id] = session
            logger.info(
                "[camera-debug] session=%s raw_attention=%s raw_mood=%s paused=%s cooldown=%d awaiting_reply=%s",
                session_id, update.attention, update.mood, session.paused_for_no_face,
                session.checkin_cooldown, session.awaiting_checkin_reply,
            )
            return session

