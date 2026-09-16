from app.llm.tutor import TutorService
from app.rag.retriever import Retriever
from app.schemas.teaching import QuizResponse
from app.teaching.session import SessionStore


class QuizService:
    def __init__(self, sessions: SessionStore, tutor: TutorService, retriever: Retriever) -> None:
        self.sessions = sessions
        self.tutor = tutor
        self.retriever = retriever

    async def generate(self, session_id: str, count: int = 5) -> QuizResponse:
        session = self.sessions.get(session_id)
        retrieval = self.retriever.retrieve(session.topic, session.course_id)
        return await self.tutor.quiz("\n".join(session.taught_content), retrieval.context, count)
