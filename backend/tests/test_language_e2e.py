import re
from pathlib import Path
from tempfile import mkdtemp
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.container import Container
from app.core.config import Settings
from app.language.detector import LanguageDetector
from app.main import create_app
from app.schemas.courses import CourseCreate
from app.storage import CourseRepository
from app.teaching.session import SessionStore
from app.teaching.notebook_lesson import NotebookLessonService


class FakeRetrieval:
    records = []
    context = "Reference context."
    source_type = "general_knowledge"
    sources = []


class FakeRetriever:
    def retrieve(self, query, course_id):
        return FakeRetrieval()


class FakeGuard:
    async def classify(self, topic):
        return SimpleNamespace(allowed=True)


class FakeGroq:
    async def json(self, messages, temperature=0.4):
        system = " ".join(
            message.get("content", "") for message in messages if message["role"] == "system"
        )
        if "Create a quiz" in system:
            return {
                "questions": [
                    {"type": "multiple_choice", "question": "Quiz question one?",
                     "options": ["a", "b", "c", "d"]}
                ]
            }
        if "Grade the student's answer" in system:
            return {"result": "correct", "explanation": "Good job."}
        return {
            "lesson_title": "Test Lesson",
            "sections": [
                {
                    "heading": "Introduction",
                    "blocks": [{"type": "text", "content": "Explaining the topic.", "latex": None}],
                    "follow_up_questions": ["Follow-up one?"],
                }
            ],
            "follow_up_questions": ["Follow-up one?"],
        }

    async def safe_call(self, messages, *, temperature=0.4, model=None, trim_history=8, retry_trim=4):
        return await self.text(messages, temperature=temperature)

    async def text(self, messages, temperature=0.6):
        system = " ".join(
            message.get("content", "") for message in messages if message["role"] == "system"
        )
        if "Based on this, ask the student ONE short" in system:
            return "Are you following along okay?"
        if "Respond entirely in Roman Urdu" in system:
            return "Aap ka sawal wazeh hai. Main aik aasan misal se samjhata hoon."
        if "Respond entirely in Urdu" in system:
            return "آپ کا سوال واضح ہے۔ میں ایک آسان مثال سے سمجھاتا ہوں۔"
        return "Spoken answer text in English."


_URDU_SCRIPT = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")


def _build_container(enabled: bool) -> Container:
    runtime_dir = Path(mkdtemp(prefix="ai-tutor-e2e-"))
    settings = Settings(language_detection_enabled=enabled)
    courses = CourseRepository(runtime_dir / "courses.json")
    course = courses.create(CourseCreate(name="E2E course"))
    course_id = course.course_id
    detector = LanguageDetector(llm=None)
    service = NotebookLessonService(
        FakeGroq(), FakeGuard(), FakeRetriever(),
        language_detector=detector,
        language_detection_enabled=enabled,
    )
    return Container(
        runtime_dir=runtime_dir, settings=settings, courses=courses,
        documents=None, ingestion=None, retriever=FakeRetriever(), guard=FakeGuard(),
        tutor=None, sessions=SessionStore(), teaching=None, followups=None,
        quizzes=None, stt=None, tts=None, vad=None, notebook_lessons=service,
        language_detector=detector,
    )


def _start_and_interrupt(client: TestClient, *, question: str) -> tuple[str, dict]:
    course_id = client.app.state.container.courses.all()[0].course_id
    start = client.post("/api/lessons/start", json={
        "course_id": course_id, "topic": "Table of 9", "language": "English",
    })
    assert start.status_code == 200
    payload = start.json()
    assert payload["response_type"] == "multiplication_table"
    interrupt = client.post(f"/api/lessons/{payload['session_id']}/interrupt", json={
        "question": question,
        "current_chunk_index": 0,
    })
    assert interrupt.status_code == 200
    return payload["session_id"], interrupt.json()


@pytest.fixture
def client(enabled):
    app = create_app()
    with TestClient(app) as test_client:
        app.state.container = _build_container(enabled)
        yield test_client


@pytest.mark.parametrize("enabled", [False])
def test_e2e_flag_off_matches_pre_feature_behavior(client):
    session_id, response = _start_and_interrupt(
        client, question="mujhe samajh nahi aya, aap kya samjhate hain"
    )
    # No language metadata, answer stays English, detection never ran.
    assert response["language"] is None
    assert response["answer"]["text"] == "Spoken answer text in English."
    assert not _URDU_SCRIPT.search(response["answer"]["text"])

    vision = client.post(f"/api/lessons/{session_id}/vision-state", json={
        "attention": "attentive", "mood": "neutral",
    })
    assert vision.status_code == 200
    assert vision.json()["language"] is None

    quiz = client.post(f"/api/lessons/{session_id}/quiz?count=3")
    assert quiz.status_code == 200
    assert quiz.json()["language"] is None


@pytest.mark.parametrize("enabled", [True])
def test_e2e_flag_on_english_turn_stays_english(client):
    session_id, response = _start_and_interrupt(
        client, question="can you please explain this step again"
    )
    assert response["language"] == "English"
    assert response["answer"]["text"] == "Spoken answer text in English."
    assert not _URDU_SCRIPT.search(response["answer"]["text"])


@pytest.mark.parametrize("enabled", [True])
def test_e2e_flag_on_roman_urdu_interrupt_answers_urdu_but_lesson_stays_english(client):
    session_id, response = _start_and_interrupt(
        client, question="mujhe samajh nahi aya, aap kya samjhate hain"
    )
    # The interrupt is answered in Roman Urdu, but the lesson keeps its original
    # English language so the lecture resumes where it left off.
    assert response["language"] == "English"
    assert response["answer_language"] == "Roman Urdu"
    assert not _URDU_SCRIPT.search(response["answer"]["text"])
    assert "Aap ka sawal" in response["answer"]["text"]

    # Subsequent vision-state reports the lesson language for voice routing.
    vision = client.post(f"/api/lessons/{session_id}/vision-state", json={
        "attention": "attentive", "mood": "neutral",
    })
    assert vision.status_code == 200
    assert vision.json()["language"] == "English"

    # Quiz + grading advertise the lesson language too.
    quiz = client.post(f"/api/lessons/{session_id}/quiz?count=3")
    assert quiz.status_code == 200
    assert quiz.json()["language"] == "English"

    grade = client.post(f"/api/lessons/{session_id}/quiz/grade", json={
        "question": {
            "id": "q1", "type": "short_answer", "question": "Quiz question one?", "options": [],
        },
        "answer": "correct answer",
    })
    assert grade.status_code == 200
    assert grade.json()["language"] == "English"