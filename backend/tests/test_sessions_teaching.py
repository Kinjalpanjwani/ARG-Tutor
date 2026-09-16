import pytest

from app.core.constants import SessionStatus
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from app.schemas.sessions import SessionCreate
from app.teaching.engine import TeachingEngine
from app.teaching.session import SessionStore
from tests.fakes import FakeEmbeddings, FakeGuard, FakePlanner, FakeTutor


def test_session_creation_has_isolated_state() -> None:
    store = SessionStore()
    first = store.create(SessionCreate(topic="Recursion"))
    second = store.create(SessionCreate(topic="Calculus"))
    first.covered_concepts.append("base case")
    assert first.session_id != second.session_id
    assert second.covered_concepts == []


@pytest.mark.asyncio
async def test_teaching_state_transitions_and_interruption(tmp_path) -> None:
    sessions = SessionStore()
    session = sessions.create(SessionCreate(topic="Recursion"))
    retriever = Retriever(FakeEmbeddings(), VectorStore(tmp_path, 4), threshold=-1)
    engine = TeachingEngine(sessions, FakeGuard(), FakePlanner(), FakeTutor(), retriever)

    first = await engine.start(session.session_id)
    assert first.content == "Teaching core idea"
    assert sessions.get(session.session_id).current_step == 1

    interrupted = await engine.respond(session.session_id, "Why a base case?", True)
    assert interrupted.next_action == "repeat_simpler"
    assert sessions.get(session.session_id).current_step == 0
    assert sessions.get(session.session_id).status == SessionStatus.TEACHING
