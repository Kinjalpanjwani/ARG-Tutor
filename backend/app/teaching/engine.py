from app.core.constants import SessionStatus
from app.llm.guardrails import AcademicGuard
from app.llm.planner import TeachingPlanner
from app.llm.tutor import TutorService, whiteboard_events
from app.rag.retriever import Retriever
from app.schemas.common import Message
from app.schemas.sessions import TeachingSession
from app.schemas.teaching import TeachingResponse
from app.teaching.session import SessionStore


class NonAcademicError(ValueError):
    pass


class InvalidSessionStateError(ValueError):
    pass


class TeachingEngine:
    """Transport-neutral lesson state machine, ready for HTTP or WebSockets."""

    def __init__(self, sessions: SessionStore, guard: AcademicGuard, planner: TeachingPlanner, tutor: TutorService, retriever: Retriever) -> None:
        self.sessions = sessions
        self.guard = guard
        self.planner = planner
        self.tutor = tutor
        self.retriever = retriever

    async def start(self, session_id: str) -> TeachingResponse:
        session = self.sessions.get(session_id)
        decision = await self.guard.classify(session.topic)
        if not decision.allowed:
            raise NonAcademicError(decision.reason)
        retrieval = self.retriever.retrieve(session.topic, session.course_id)
        session.lesson_plan = await self.planner.create_plan(
            session.topic, session.student_level.value, session.language, retrieval.context
        )
        session.retrieved_sources = retrieval.sources
        session.status = SessionStatus.TEACHING
        self.sessions.save(session)
        return await self.next(session_id)

    async def next(self, session_id: str) -> TeachingResponse:
        session = self.sessions.get(session_id)
        if session.status in {SessionStatus.PAUSED, SessionStatus.COMPLETED}:
            raise InvalidSessionStateError(f"Cannot advance a {session.status} session")
        if not session.lesson_plan:
            raise InvalidSessionStateError("Session has not been planned")
        if session.current_step >= len(session.lesson_plan.steps):
            session.status = SessionStatus.COMPLETED
            self.sessions.save(session)
            return TeachingResponse(
                session_id=session_id, type="lesson_complete", content="Lesson complete.",
                source_type="course_material" if session.retrieved_sources else "general_knowledge",
                expects_student_response=False, session_status=session.status,
            )
        step = session.lesson_plan.steps[session.current_step]
        retrieval = self.retriever.retrieve(step.goal, session.course_id)
        data = await self.tutor.teach_step(
            topic=session.topic, goal=step.goal, step_type=step.type,
            level=session.student_level.value, language=session.language,
            context=retrieval.context,
            history="\n".join(item.content for item in session.conversation_history[-6:]),
        )
        content = str(data.get("content", "")).strip()
        if not content:
            raise InvalidSessionStateError("Tutor produced an empty teaching step")
        expects = bool(data.get("expects_student_response", step.type == "check"))
        session.taught_content.append(content)
        session.conversation_history.append(Message(role="assistant", content=content))
        session.covered_concepts.append(step.goal)
        session.current_step += 1
        session.response_revision += 1
        session.retrieved_sources = retrieval.sources
        session.status = SessionStatus.WAITING if expects else SessionStatus.TEACHING
        self.sessions.save(session)
        return TeachingResponse(
            session_id=session_id, type="teaching_step", content=content,
            whiteboard=whiteboard_events(data.get("whiteboard"), content),
            source_type=retrieval.source_type, sources=retrieval.sources,
            expects_student_response=expects, session_status=session.status,
        )

    async def respond(self, session_id: str, content: str, is_interruption: bool) -> TeachingResponse:
        session = self.sessions.get(session_id)
        decision = await self.guard.classify(
            f"In an active academic lesson about {session.topic}, the student says: {content}"
        )
        if not decision.allowed:
            raise NonAcademicError(decision.reason)
        session.conversation_history.append(Message(role="user", content=content))
        session.status = SessionStatus.ANSWERING
        session.response_revision += 1  # clients can invalidate audio from an older revision
        plan = session.lesson_plan
        goal = plan.steps[max(0, session.current_step - 1)].goal if plan else session.topic
        retrieval = self.retriever.retrieve(content, session.course_id)
        history = "\n".join(item.content for item in session.conversation_history[-6:])
        if is_interruption:
            data = await self.tutor.interruption(
                question=content, level=session.student_level.value, language=session.language,
                current_goal=goal, context=retrieval.context, history=history,
            )
        else:
            data = await self.tutor.evaluate_feedback(
                response=content, level=session.student_level.value,
                language=session.language, current_goal=goal, history=history,
            )
        answer = str(data.get("content", "")).strip()
        next_action = str(data.get("next_action", "resume_same_step"))
        unclear = data.get("unclear_concept")
        if unclear and str(unclear) not in session.unclear_concepts:
            session.unclear_concepts.append(str(unclear))
        if next_action in {"repeat_simpler", "give_example", "go_deeper"}:
            session.current_step = max(0, session.current_step - 1)
        session.conversation_history.append(Message(role="assistant", content=answer))
        session.status = SessionStatus.TEACHING
        session.retrieved_sources = retrieval.sources
        self.sessions.save(session)
        return TeachingResponse(
            session_id=session_id,
            type="interruption_answer" if is_interruption else "teaching_step",
            content=answer,
            whiteboard=whiteboard_events(data.get("whiteboard"), answer),
            source_type=retrieval.source_type, sources=retrieval.sources,
            expects_student_response=False, session_status=session.status,
            next_action=next_action,
        )
