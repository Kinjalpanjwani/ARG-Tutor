import pytest
from fastapi.testclient import TestClient

from app.core.constants import AttentionState, MoodState, SessionStatus
from app.main import create_app
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from app.schemas.sessions import SessionCreate
from app.teaching.engine import TeachingEngine
from app.teaching.session import SessionStore
from app.teaching.vision import (
    build_checkin_followup_prompt,
    build_checkin_prompt,
    tone_directive_for_state,
)
from tests.fakes import FakeEmbeddings, FakeGuard, FakePlanner, FakeTutor


@pytest.fixture
def app_client():
    app = create_app()
    with TestClient(app) as client:
        yield client



def test_tone_directive_for_state():
    assert tone_directive_for_state(AttentionState.ATTENTIVE, MoodState.NEUTRAL) is None
    assert tone_directive_for_state(None, None) is None

    # Individual attention states
    distracted_dir = tone_directive_for_state(AttentionState.DISTRACTED, MoodState.NEUTRAL)
    assert "looking away" in distracted_dir
    assert "re-engaging" in distracted_dir

    drowsy_dir = tone_directive_for_state(AttentionState.DROWSY, MoodState.NEUTRAL)
    assert "sleepy" in drowsy_dir
    assert "more energy" in drowsy_dir

    # Individual mood states
    confused_dir = tone_directive_for_state(AttentionState.ATTENTIVE, MoodState.CONFUSED)
    assert "confused" in confused_dir
    assert "simpler example" in confused_dir

    happy_dir = tone_directive_for_state(AttentionState.ATTENTIVE, MoodState.HAPPY)
    assert happy_dir is not None
    assert "happy" in happy_dir.lower()

    stressed_dir = tone_directive_for_state(AttentionState.ATTENTIVE, MoodState.STRESSED)
    assert "tense" in stressed_dir
    assert "extra gentle" in stressed_dir

    sad_dir = tone_directive_for_state(AttentionState.ATTENTIVE, MoodState.SAD)
    assert "down" in sad_dir
    assert "warm and encouraging" in sad_dir

    # Combined state
    combined_dir = tone_directive_for_state(AttentionState.DISTRACTED, MoodState.CONFUSED)
    assert "looking away" in combined_dir
    assert "confused" in combined_dir


def test_build_checkin_prompts():
    directive = "The student looks confused -- slow down."
    q_prompt = build_checkin_prompt(directive)
    assert "check-in question" in q_prompt
    assert directive in q_prompt

    follow_prompt = build_checkin_followup_prompt(
        question="Would you like a simpler example?",
        directive=directive,
        reply="Yes please, with numbers",
        last_chunk="Matrix multiplication is defined as...",
    )
    assert "Yes please, with numbers" in follow_prompt
    assert "Matrix multiplication" in follow_prompt
    assert "acknowledge" in follow_prompt


def test_vision_state_endpoint(app_client):
    # 1. Create a session
    resp = app_client.post("/api/sessions", json={"topic": "Physics forces", "student_level": "secondary"})
    assert resp.status_code == 201
    session_id = resp.json()["session_id"]

    # 2. Update vision state to attentive / neutral
    v_resp = app_client.post(f"/api/sessions/{session_id}/vision-state", json={
        "attention": "attentive",
        "mood": "neutral"
    })
    assert v_resp.status_code == 200
    data = v_resp.json()
    assert data["session_id"] == session_id
    assert data["attention"] == "attentive"
    assert data["mood"] == "neutral"
    assert data["paused"] is False

    # 3. Verify on non-existent session
    bad_resp = app_client.post("/api/sessions/non-existent-id/vision-state", json={
        "attention": "attentive",
        "mood": "neutral"
    })
    assert bad_resp.status_code == 404


def test_no_face_pause_and_resume_flow(app_client):
    resp = app_client.post("/api/sessions", json={"topic": "Biology cells", "student_level": "secondary"})
    session_id = resp.json()["session_id"]

    # Push 1: no_face
    v1 = app_client.post(f"/api/sessions/{session_id}/vision-state", json={"attention": "no_face", "mood": "neutral"})
    assert v1.json()["paused"] is False

    # Push 2: no_face
    v2 = app_client.post(f"/api/sessions/{session_id}/vision-state", json={"attention": "no_face", "mood": "neutral"})
    assert v2.json()["paused"] is False

    # Push 3: no_face -> streak reached (3), session should pause!
    v3 = app_client.post(f"/api/sessions/{session_id}/vision-state", json={"attention": "no_face", "mood": "neutral"})
    assert v3.json()["paused"] is True

    # Next call to teaching/next should return pause notice
    next_resp = app_client.post(f"/api/teaching/{session_id}/next")
    assert next_resp.status_code == 200
    next_data = next_resp.json()
    assert next_data["interrupt_type"] == "no_face_pause"
    assert "Pausing until you're back" in next_data["content"]
    assert next_data["session_status"] == "paused"

    # Push 4: face returns!
    v4 = app_client.post(f"/api/sessions/{session_id}/vision-state", json={"attention": "attentive", "mood": "neutral"})
    assert v4.json()["paused"] is False

    # Next call to teaching/next should return welcome back notice
    resume_resp = app_client.post(f"/api/teaching/{session_id}/next")
    assert resume_resp.status_code == 200
    resume_data = resume_resp.json()
    assert "Welcome back" in resume_data["content"]


@pytest.mark.asyncio
async def test_vision_checkin_and_adaptation_flow(tmp_path):
    sessions = SessionStore()
    session = sessions.create(SessionCreate(topic="Calculus Derivatives"))
    retriever = Retriever(FakeEmbeddings(), VectorStore(tmp_path, 4), threshold=-1)
    engine = TeachingEngine(sessions, FakeGuard(), FakePlanner(), FakeTutor(), retriever)

    # Start the session
    first = await engine.start(session.session_id)
    assert first.content == "Teaching core idea"

    # Now student becomes confused
    from app.schemas.vision import VisionStateUpdate
    sessions.update_vision_state(session.session_id, VisionStateUpdate(attention=AttentionState.ATTENTIVE, mood=MoodState.CONFUSED))

    # Next step should trigger a check-in question
    checkin_response = await engine.next(session.session_id)
    assert checkin_response.interrupt_type == "checkin"
    assert checkin_response.expects_student_response is True
    assert sessions.get(session.session_id).awaiting_checkin_reply is True
    assert sessions.get(session.session_id).checkin_cooldown == 3

    # Student replies to the checkin question
    adapted_response = await engine.respond(session.session_id, "Yes, please explain it simpler", is_interruption=False)
    assert adapted_response.interrupt_type == "checkin"
    assert "easier way to think about it" in adapted_response.content
    assert sessions.get(session.session_id).awaiting_checkin_reply is False


@pytest.mark.asyncio
async def test_speech_transcribe_direct_and_fallback(tmp_path, monkeypatch):
    from pathlib import Path
    from app.speech.stt import SpeechToTextService

    class FakeGroqSTT:
        def __init__(self):
            self.calls = []

        async def transcribe(self, path: str):
            self.calls.append(path)
            if path.endswith(".raw"):
                raise RuntimeError("Raw audio cannot be decoded by Groq direct")
            return "This is transcribed speech"

    class FakeVAD:
        def contains_speech(self, path: Path):
            return True

    fake_groq = FakeGroqSTT()
    stt_service = SpeechToTextService(fake_groq, FakeVAD())

    # 1. Direct success
    success_file = tmp_path / "test.webm"
    success_file.write_bytes(b"dummy audio data")
    has_speech, text = await stt_service.transcribe(success_file)
    assert has_speech is True
    assert text == "This is transcribed speech"
    assert len(fake_groq.calls) == 1

    # 2. Direct failure with fallback to ffmpeg
    raw_file = tmp_path / "test.raw"
    raw_file.write_bytes(b"raw data")

    # Mock _get_ffmpeg_executable and subprocess.run to simulate successful ffmpeg conversion
    monkeypatch.setattr("app.speech.stt._get_ffmpeg_executable", lambda: "ffmpeg")
    
    class FakeProcess:
        returncode = 0
        stderr = b""

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: FakeProcess())

    has_speech2, text2 = await stt_service.transcribe(raw_file)
    assert has_speech2 is True
    assert text2 == "This is transcribed speech"
    # First was test.raw (which failed), second was the temporary .wav converted by ffmpeg
    assert len(fake_groq.calls) == 3
    assert fake_groq.calls[1].endswith(".raw")
    assert fake_groq.calls[2].endswith(".wav")


@pytest.mark.asyncio
async def test_notebook_lesson_next_section_gating(tmp_path):
    from app.teaching.notebook_lesson import NotebookLessonService, NotebookLesson
    from app.schemas.lesson import WhiteboardBlock

    class FakeLLMNotebook:
        async def text(self, messages, temperature=0.6):
            return "How are you finding this topic so far?"

    retriever = Retriever(FakeEmbeddings(), VectorStore(tmp_path, 4), threshold=-1)
    service = NotebookLessonService(FakeLLMNotebook(), FakeGuard(), retriever)

    session_id = "test-nb-session"
    lesson = NotebookLesson(
        session_id=session_id,
        course_id="c",
        topic="Biology",
        language="English",
        lecture_text="Initial lecture",
        chunks=[WhiteboardBlock(id="b1", type="text", text="Initial chunk")],
        sources=[],
        pending_sections=[("Cell Division", "Details about mitosis...")],
    )
    service.store.put(lesson)

    # 1. Normal next_section call (no vision flags triggered)
    # Checkin cooldown is initially 0, but attention is attentive and mood is neutral
    assert lesson.latest_attention == "attentive"
    assert lesson.latest_mood == "neutral"

    # 2. Simulate no-face pause
    from app.schemas.vision import VisionStateUpdate
    service.store.update_vision_state(session_id, VisionStateUpdate(attention=AttentionState.NO_FACE, mood=MoodState.NEUTRAL))
    service.store.update_vision_state(session_id, VisionStateUpdate(attention=AttentionState.NO_FACE, mood=MoodState.NEUTRAL))
    service.store.update_vision_state(session_id, VisionStateUpdate(attention=AttentionState.NO_FACE, mood=MoodState.NEUTRAL))
    assert service.store.get(session_id).paused_for_no_face is True

    sec_paused = await service.next_section(session_id)
    assert sec_paused.interrupt_type == "no_face_pause"
    assert sec_paused.chunks[0].type == "no_face_pause"

    # 3. Simulate face returning
    service.store.update_vision_state(session_id, VisionStateUpdate(attention=AttentionState.ATTENTIVE, mood=MoodState.NEUTRAL))
    assert service.store.get(session_id).paused_for_no_face is False

    sec_resumed = await service.next_section(session_id)
    assert sec_resumed.chunks[0].type == "note"
    assert "Welcome back" in sec_resumed.chunks[0].text

    # 4. Simulate confused student -> triggers check-in
    service.store.update_vision_state(session_id, VisionStateUpdate(attention=AttentionState.ATTENTIVE, mood=MoodState.CONFUSED))
    sec_checkin = await service.next_section(session_id)
    assert sec_checkin.interrupt_type == "checkin"
    assert sec_checkin.chunks[0].type == "checkin"
    assert "How are you finding this topic" in sec_checkin.chunks[0].text
    assert service.store.get(session_id).awaiting_checkin_reply is True


@pytest.mark.asyncio
async def test_notebook_checkin_blocks_next_section_until_cancelled(tmp_path):
    from app.teaching.notebook_lesson import NotebookLessonService, NotebookLesson
    from app.schemas.lesson import WhiteboardBlock
    from app.schemas.vision import VisionStateUpdate
    from app.core.constants import AttentionState

    class FakeLLMCheckin:
        async def text(self, messages, temperature=0.6):
            return "Do you want to slow down?"

    retriever = Retriever(FakeEmbeddings(), VectorStore(tmp_path, 4), threshold=-1)
    service = NotebookLessonService(FakeLLMCheckin(), FakeGuard(), retriever)
    session_id = "test-checkin-cancel"
    lesson = NotebookLesson(
        session_id=session_id, course_id="c", topic="Physics", language="English",
        lecture_text="L", chunks=[WhiteboardBlock(id="b1", type="text", text="Chunk one")],
        sources=[], pending_sections=[("Next", "context")],
    )
    service.store.put(lesson)

    # Push-fired check-in while distracted
    service.store.update_vision_state(session_id, VisionStateUpdate(attention=AttentionState.DISTRACTED, mood=MoodState.NEUTRAL))

    question = await service.maybe_fire_checkin(session_id)
    assert question == "Do you want to slow down?"
    assert service.store.get(session_id).awaiting_checkin_reply is True

    # While a reply is awaited, next_section serves nothing and consumes no chunks.
    blocked = await service.next_section(session_id)
    assert blocked.chunks == []
    assert service.store.get(session_id).chunks[0].id == "b1"

    # Cancelling clears the pending state so the lecture can continue.
    service.cancel_checkin(session_id)
    assert service.store.get(session_id).awaiting_checkin_reply is False

    resumed = await service.next_section(session_id)
    assert resumed.chunks[0].id == "b1"
