from app.core.constants import (
    AttentionState,
    MoodState,
    SessionStatus,
    VISION_CHECKIN_COOLDOWN_STEPS,
)
from app.llm.guardrails import AcademicGuard
from app.llm.planner import TeachingPlanner
from app.llm.tutor import TutorService, whiteboard_events
from app.rag.retriever import Retriever
from app.schemas.common import Message, WhiteboardEvent
from app.schemas.sessions import TeachingSession
from app.schemas.teaching import TeachingResponse
from app.teaching.session import SessionStore
from app.teaching.vision import tone_directive_for_state


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

        # 1. No-face pause handling
        if session.paused_for_no_face or (session.status == SessionStatus.PAUSED and session.paused_for_no_face):
            return TeachingResponse(
                session_id=session_id,
                type="teaching_step",
                content="Still there? Pausing until you're back.",
                whiteboard=[WhiteboardEvent(content="Still there? Pausing until you're back.", format="text")],
                source_type="general_knowledge",
                expects_student_response=False,
                session_status=SessionStatus.PAUSED,
                interrupt_type="no_face_pause",
            )

        # 2. Resumed from no-face notification
        if session.resumed_from_no_face:
            session.resumed_from_no_face = False
            self.sessions.save(session)
            return TeachingResponse(
                session_id=session_id,
                type="teaching_step",
                content="Welcome back — let's pick up where we left off.",
                whiteboard=[WhiteboardEvent(content="Welcome back — let's pick up where we left off.", format="text")],
                source_type="general_knowledge",
                expects_student_response=False,
                session_status=session.status,
                interrupt_type=None,
            )

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

        tone_directive = tone_directive_for_state(session.latest_attention, session.latest_mood)

        # 3. Camera check-in question check
        needs_checkin = (
            session.checkin_cooldown <= 0
            and tone_directive is not None
            and (
                session.latest_attention in {AttentionState.DISTRACTED, AttentionState.DROWSY}
                or session.latest_mood in {MoodState.CONFUSED, MoodState.STRESSED, MoodState.SAD}
            )
        )

        if needs_checkin:
            try:
                question = await self.tutor.generate_checkin(
                    directive=tone_directive,
                    level=session.student_level.value,
                    language=session.language,
                )
            except Exception:
                question = "Are you following along okay, or would you prefer a simpler example?"

            session.awaiting_checkin_reply = True
            session.checkin_cooldown = VISION_CHECKIN_COOLDOWN_STEPS
            session.last_checkin_question = question
            session.last_checkin_directive = tone_directive
            session.conversation_history.append(Message(role="assistant", content=question))
            session.status = SessionStatus.WAITING
            self.sessions.save(session)
            return TeachingResponse(
                session_id=session_id,
                type="teaching_step",
                content=question,
                whiteboard=[WhiteboardEvent(content=question, format="question")],
                source_type="general_knowledge",
                sources=[],
                expects_student_response=True,
                session_status=session.status,
                interrupt_type="checkin",
            )

        session.checkin_cooldown = max(0, session.checkin_cooldown - 1)

        step = session.lesson_plan.steps[session.current_step]
        retrieval = self.retriever.retrieve(f"{session.topic}: {step.goal}", session.course_id)
        remaining_goals = [item.goal for item in session.lesson_plan.steps[session.current_step + 1:]]
        lesson_state = (
            f"Lesson topic: {session.topic}\n"
            f"Learning objective: {session.lesson_plan.learning_objective}\n"
            f"Concepts already taught: {session.covered_concepts}\n"
            f"Current goal: {step.goal}\n"
            f"Goals still to teach: {remaining_goals}\n"
            f"Concepts the student found unclear: {session.unclear_concepts}\n"
            f"Recent explanations:\n" + "\n".join(session.taught_content[-4:])
        )
        data = await self.tutor.teach_step(
            topic=session.topic, goal=step.goal, step_type=step.type,
            level=session.student_level.value, language=session.language,
            context=retrieval.context,
            history="\n".join(f"{item.role}: {item.content}" for item in session.conversation_history[-8:]),
            lesson_state=lesson_state,
            tone_directive=tone_directive,
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

        # Check if replying to a vision check-in question
        if session.awaiting_checkin_reply:
            session.awaiting_checkin_reply = False
            last_chunk = (
                session.taught_content[-1]
                if session.taught_content
                else (session.lesson_plan.steps[0].goal if session.lesson_plan else session.topic)
            )
            history = "\n".join(item.content for item in session.conversation_history[-6:])
            directive = session.last_checkin_directive or "The student needed an adapted explanation."
            question = session.last_checkin_question or "How can I help clarify?"
            try:
                answer = await self.tutor.generate_checkin_followup(
                    question=question,
                    directive=directive,
                    reply=content,
                    last_chunk=last_chunk,
                    level=session.student_level.value,
                    language=session.language,
                    history=history,
                )
            except Exception:
                answer = "Okay, let's take that a bit more gently -- and let's get back to the lesson."

            session.conversation_history.append(Message(role="assistant", content=answer))
            session.status = SessionStatus.TEACHING
            self.sessions.save(session)
            return TeachingResponse(
                session_id=session_id,
                type="interruption_answer" if is_interruption else "teaching_step",
                content=answer,
                whiteboard=whiteboard_events(None, answer),
                source_type="general_knowledge",
                sources=[],
                expects_student_response=False,
                session_status=session.status,
                next_action="resume_same_step",
                interrupt_type="checkin",
            )

        plan = session.lesson_plan
        goal = plan.steps[max(0, session.current_step - 1)].goal if plan else session.topic
        retrieval = self.retriever.retrieve(content, session.course_id)
        history = "\n".join(item.content for item in session.conversation_history[-6:])
        tone_directive = tone_directive_for_state(session.latest_attention, session.latest_mood)

        if is_interruption:
            data = await self.tutor.interruption(
                question=content, level=session.student_level.value, language=session.language,
                current_goal=goal, context=retrieval.context, history=history,
                tone_directive=tone_directive,
            )
        else:
            data = await self.tutor.evaluate_feedback(
                response=content, level=session.student_level.value,
                language=session.language, current_goal=goal, history=history,
                tone_directive=tone_directive,
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

