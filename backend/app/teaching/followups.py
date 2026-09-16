from app.llm.tutor import TutorService
from app.teaching.session import SessionStore


class FollowupService:
    def __init__(self, sessions: SessionStore, tutor: TutorService) -> None:
        self.sessions = sessions
        self.tutor = tutor

    async def generate(self, session_id: str, count: int = 3) -> list[str]:
        session = self.sessions.get(session_id)
        return await self.tutor.followups(
            "\n".join(session.taught_content), session.student_level.value,
            session.covered_concepts, session.unclear_concepts, count,
        )
