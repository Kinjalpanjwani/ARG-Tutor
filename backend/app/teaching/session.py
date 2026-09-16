from datetime import datetime, timezone
from threading import RLock

from app.schemas.sessions import SessionCreate, TeachingSession


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
