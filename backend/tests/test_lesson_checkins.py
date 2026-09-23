import pytest

from app.schemas.lesson import WhiteboardBlock
from app.teaching.notebook_lesson import (
    CONTENT_CHECKIN_DIRECTIVE,
    NotebookLesson,
    NotebookLessonService,
)


class _FakeLLM:
    async def json(self, messages, temperature=0.4):
        raise AssertionError("json should not be called in these tests")

    async def text(self, messages, temperature=0.6):
        return "Spoken answer text"

    async def safe_call(self, messages, *, temperature=0.4, model=None, trim_history=8, retry_trim=4):
        return await self.text(messages, temperature=temperature)


class _FakeGuard:
    async def classify(self, topic):
        return type("Decision", (), {"allowed": True})()


class _FakeRetrieval:
    context = "Ref context."
    source_type = "general_knowledge"
    sources = []


class _FakeRetriever:
    def retrieve(self, query, course_id):
        return _FakeRetrieval()


def _service():
    return NotebookLessonService(_FakeLLM(), _FakeGuard(), _FakeRetriever())


def test_structured_blocks_keeps_checkin_type():
    payload = {
        "lesson_title": "Tables",
        "sections": [
            {
                "heading": "Table of 2",
                "blocks": [
                    {"type": "text", "content": "Let's learn the table of 2."},
                    {"type": "checkin", "content": "Now you tell me: what is 7 × 2?"},
                ],
            }
        ],
        "follow_up_questions": [],
    }
    blocks, _ = NotebookLessonService._structured_blocks(payload)
    assert blocks[1].type == "text"
    assert blocks[2].type == "checkin"
    assert blocks[2].text == "Now you tell me: what is 7 × 2?"


@pytest.mark.asyncio
async def test_next_section_converts_scheduled_checkin_to_interaction():
    service = _service()
    lesson = NotebookLesson(
        session_id="s", course_id="c", topic="Tables", language="English",
        lecture_text="Lecture",
        chunks=[
            WhiteboardBlock(id="b1", type="text", text="Line one."),
            WhiteboardBlock(id="q1", type="checkin", text="What is 7 × 2?"),
            WhiteboardBlock(id="b2", type="text", text="Line three."),
        ],
        sources=[],
    )
    service.store.put(lesson)

    first = await service.next_section("s")
    assert first.chunks[0].id == "b1"

    checkin = await service.next_section("s")
    assert checkin.interrupt_type == "checkin"
    assert checkin.chunks[0].type == "checkin"
    assert checkin.chunks[0].text == "What is 7 × 2?"
    assert lesson.awaiting_checkin_reply is True
    assert lesson.last_checkin_question == "What is 7 × 2?"
    assert lesson.last_checkin_directive == CONTENT_CHECKIN_DIRECTIVE

    blocked = await service.next_section("s")
    assert blocked.chunks == []


@pytest.mark.asyncio
async def test_interrupt_reply_to_scheduled_checkin_resumes_and_clears():
    service = _service()
    lesson = NotebookLesson(
        session_id="s", course_id="c", topic="Tables", language="English",
        lecture_text="Lecture",
        chunks=[
            WhiteboardBlock(id="b1", type="text", text="Line one."),
            WhiteboardBlock(id="q1", type="checkin", text="What is 7 × 2?"),
            WhiteboardBlock(id="b2", type="text", text="Line three."),
        ],
        sources=[],
    )
    service.store.put(lesson)
    await service.next_section("s")  # serve b1
    await service.next_section("s")  # serve checkin, arm reply
    assert lesson.awaiting_checkin_reply is True

    response = await service.interrupt("s", "14", 0)
    assert lesson.awaiting_checkin_reply is False
    assert response.interrupt_type == "checkin"
    assert response.answer.interrupt_type == "checkin"
    assert response.resume_chunk_index == 1
    assert response.answer.text == "Spoken answer text"


@pytest.mark.asyncio
async def test_table_embedded_checkin_reply_uses_checkin_flow():
    service = _service()
    kind, blocks, _ = NotebookLessonService._table_fast_path("table of 2?", "English")
    lesson = NotebookLesson(
        session_id="s", course_id="c", topic="Table of 2", language="English",
        lecture_text=" ".join(block.text for block in blocks),
        chunks=blocks, sources=[], response_type=kind,
    )
    service.store.put(lesson)
    checkin_index = [i for i, block in enumerate(blocks) if block.type == "checkin"][0]
    assert checkin_index == 6

    response = await service.interrupt("s", "14", checkin_index)
    assert response.interrupt_type == "checkin"
    assert response.answer.interrupt_type == "checkin"
    assert response.resume_chunk_index == checkin_index + 1
    assert lesson.awaiting_checkin_reply is False


@pytest.mark.asyncio
async def test_table_interrupt_at_normal_row_uses_general_branch():
    service = _service()
    kind, blocks, _ = NotebookLessonService._table_fast_path("table of 2?", "English")
    lesson = NotebookLesson(
        session_id="s", course_id="c", topic="Table of 2", language="English",
        lecture_text=" ".join(block.text for block in blocks),
        chunks=blocks, sources=[], response_type=kind,
    )
    service.store.put(lesson)

    response = await service.interrupt("s", "can you go slower?", 1)
    assert response.interrupt_type is None
    assert response.resume_chunk_index == 2


@pytest.mark.asyncio
async def test_teach_again_reexplains_latest_taught_concept_after_lesson_end():
    service = _service()
    lesson = NotebookLesson(
        session_id="s", course_id="c", topic="Photosynthesis", language="English",
        lecture_text="Plants convert light energy into stored chemical energy.",
        chunks=[], sources=[], active_section="Energy conversion",
        delivered_blocks=[
            WhiteboardBlock(id="b1", type="text", text="Chlorophyll absorbs light energy."),
            WhiteboardBlock(id="q1", type="checkin", text="What does chlorophyll absorb?"),
        ],
    )
    service.store.put(lesson)

    response = await service.interrupt("s", "Teach me again", 1)

    assert response.reteach is True
    assert response.answer.text == "Spoken answer text"
    assert response.resume_chunk_index == 2
    assert lesson.history[-2]["content"] == "Teach me again"


@pytest.mark.parametrize("student_request", [
    "Explain that again",
    "I didn't understand",
    "Go over it again",
    "Repeat this concept",
])
def test_reteach_request_detection(student_request):
    assert NotebookLessonService._is_reteach_request(student_request) is True
