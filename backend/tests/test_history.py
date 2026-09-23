from pathlib import Path
from tempfile import mkdtemp
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.container import Container
from app.core.config import Settings
from app.main import create_app
from app.schemas.courses import CourseCreate
from app.schemas.lesson import LessonRecord
from app.storage import CourseRepository, JsonRepository
from app.teaching.archive import LessonArchiveService
from app.teaching.flashcards import FlashcardService
from app.teaching.notebook_lesson import NotebookLessonService
from app.teaching.session import SessionStore


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
        if "flashcard" in system.lower():
            return {
                "cards": [
                    {"front": "Term one", "back": "Explanation one."},
                    {"front": "Term two", "back": "Explanation two."},
                ]
            }
        if "Create a quiz" in system:
            return {
                "questions": [
                    {"type": "multiple_choice", "question": "Quiz question one?",
                     "options": ["a", "b", "c", "d"]}
                ]
            }
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
        return "Spoken answer text in English."

    async def text(self, messages, temperature=0.6):
        return "Spoken answer text in English."


def _build_container() -> Container:
    runtime_dir = Path(mkdtemp(prefix="ai-tutor-history-"))
    settings = Settings(language_detection_enabled=False)
    courses = CourseRepository(runtime_dir / "courses.json")
    courses.create(CourseCreate(name="History course"))
    service = NotebookLessonService(FakeGroq(), FakeGuard(), FakeRetriever())
    return Container(
        runtime_dir=runtime_dir, settings=settings, courses=courses,
        documents=None, ingestion=None, retriever=FakeRetriever(), guard=FakeGuard(),
        tutor=None, sessions=SessionStore(), teaching=None, followups=None,
        quizzes=None, stt=None, tts=None, vad=None, notebook_lessons=service,
        history=LessonArchiveService(service.store, JsonRepository(runtime_dir / "lessons.json", LessonRecord)),
        flashcards=FlashcardService(FakeGroq(), service.store),
        language_detector=None,
    )


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as test_client:
        app.state.container = _build_container()
        yield test_client


def _start_lesson(client: TestClient) -> str:
    course_id = client.app.state.container.courses.all()[0].course_id
    start = client.post("/api/lessons/start", json={
        "course_id": course_id, "topic": "Table of 9", "language": "English",
    })
    assert start.status_code == 200
    return start.json()["session_id"]


def test_archive_then_history_lists_and_returns_record(client):
    session_id = _start_lesson(client)
    archive = client.post(f"/api/lessons/{session_id}/archive", json={
        "blocks": [
            {"type": "heading", "text": "Introduction"},
            {"type": "text", "text": "Explaining the topic."},
        ]
    })
    assert archive.status_code == 200
    assert archive.json()["session_id"] == session_id

    history = client.get("/api/lessons/history")
    assert history.status_code == 200
    entries = history.json()
    assert len(entries) == 1
    assert entries[0]["session_id"] == session_id
    assert entries[0]["topic"] == "Table of 9"
    assert entries[0]["block_count"] == 2

    record = client.get(f"/api/lessons/history/{session_id}")
    assert record.status_code == 200
    assert record.json()["blocks"][1]["text"] == "Explaining the topic."


def test_history_filters_by_course(client):
    session_id = _start_lesson(client)
    client.post(f"/api/lessons/{session_id}/archive", json={
        "blocks": [{"type": "text", "text": "Only block."}]
    })
    course_id = client.app.state.container.courses.all()[0].course_id
    matched = client.get(f"/api/lessons/history?course_id={course_id}")
    assert matched.status_code == 200
    assert len(matched.json()) == 1
    missing = client.get("/api/lessons/history?course_id=does-not-exist")
    assert len(missing.json()) == 0


def test_archive_is_idempotent_upsert(client):
    session_id = _start_lesson(client)
    for _ in range(2):
        response = client.post(f"/api/lessons/{session_id}/archive", json={
            "blocks": [{"type": "text", "text": "Latest block."}]
        })
        assert response.status_code == 200
    entries = client.get("/api/lessons/history").json()
    assert len(entries) == 1


def test_quiz_result_updates_archived_record(client):
    session_id = _start_lesson(client)
    archived = client.post(f"/api/lessons/{session_id}/archive", json={
        "blocks": [{"type": "text", "text": "Only block."}]
    })
    assert archived.status_code == 200

    quiz = client.post(f"/api/lessons/{session_id}/quiz-result", json={
        "score": 3,
        "total": 5,
        "results": [
            {"question": "Question one?", "answer": "x", "result": "correct", "explanation": "Good."},
            {"question": "Question two?", "answer": "y", "result": "incorrect", "explanation": "Nope."},
        ],
    })
    assert quiz.status_code == 200
    assert quiz.json()["quiz_score"] == 3
    assert quiz.json()["quiz_total"] == 5

    entries = client.get("/api/lessons/history").json()
    assert entries[0]["quiz_score"] == 3
    assert entries[0]["quiz_total"] == 5
    record = client.get(f"/api/lessons/history/{session_id}").json()
    assert len(record["quiz_results"]) == 2


def test_quiz_result_without_prior_archive_snapshots_lesson(client):
    session_id = _start_lesson(client)
    quiz = client.post(f"/api/lessons/{session_id}/quiz-result", json={
        "score": 1, "total": 1,
        "results": [{"question": "Q?", "answer": "a", "result": "correct", "explanation": "Ok."}],
    })
    assert quiz.status_code == 200
    record = client.get(f"/api/lessons/history/{session_id}").json()
    assert record["topic"] == "Table of 9"
    assert record["quiz_score"] == 1
    assert record["blocks"]


def test_flashcards_generate_from_lesson(client):
    session_id = _start_lesson(client)
    response = client.post(f"/api/lessons/{session_id}/flashcards?count=5")
    assert response.status_code == 200
    cards = response.json()["cards"]
    assert len(cards) == 2
    assert cards[0]["front"] == "Term one"
    assert cards[0]["back"] == "Explanation one."


def test_flashcards_advertise_language_for_roman_urdu(client):
    course_id = client.app.state.container.courses.all()[0].course_id
    start = client.post("/api/lessons/start", json={
        "course_id": course_id, "topic": "Table of 9", "language": "Roman Urdu",
    })
    assert start.status_code == 200
    session_id = start.json()["session_id"]
    response = client.post(f"/api/lessons/{session_id}/flashcards?count=5")
    assert response.status_code == 200
    body = response.json()
    assert body["language"] == "Roman Urdu"
    assert body["cards"]


def test_flashcards_count_out_of_range_rejected(client):
    session_id = _start_lesson(client)
    assert client.post(f"/api/lessons/{session_id}/flashcards?count=25").status_code == 422


def test_flashcards_pdf_generates_download(client):
    session_id = _start_lesson(client)
    response = client.get(f"/api/lessons/{session_id}/flashcards/pdf")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    disposition = response.headers.get("content-disposition", "")
    assert "attachment" in disposition
    assert "flashcards-table-of-9.pdf" in disposition
    body = response.content
    assert body.startswith(b"%PDF-")
    assert len(body) > 500


def test_flashcards_pdf_unknown_session_returns_404(client):
    assert client.get("/api/lessons/nope/flashcards/pdf").status_code == 404


def test_unknown_session_returns_404(client):
    assert client.get("/api/lessons/history/nope").status_code == 404
    assert client.post("/api/lessons/nope/archive", json={
        "blocks": [{"type": "text", "text": "x"}]
    }).status_code == 404
    assert client.post("/api/lessons/nope/quiz-result", json={
        "score": 1, "total": 1, "results": []
    }).status_code == 404
    assert client.post("/api/lessons/nope/flashcards").status_code == 404