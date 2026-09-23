import pytest

from app.language.detector import LanguageDetector
from app.language.roman_urdu_lexicon import ENGLISH_STOPWORDS
from app.schemas.lesson import WhiteboardBlock
from app.teaching.notebook_lesson import NotebookLesson, NotebookLessonService
from tests.fakes import FakeGuard


class _FakeLLM:
    async def json(self, messages, temperature=0.4):
        raise AssertionError("json should not be called in these tests")

    async def text(self, messages, temperature=0.6):
        return "Spoken answer text"

    async def safe_call(self, messages, *, temperature=0.4, model=None, trim_history=8, retry_trim=4):
        return await self.text(messages, temperature=temperature)


class _FakeRetrieval:
    context = "Reference context."
    source_type = "general_knowledge"
    sources = []


class _FakeRetriever:
    def retrieve(self, query, course_id):
        return _FakeRetrieval()


class _BombDetector:
    """Raises if invoked — used to prove the flag short-circuits before analyze()."""

    async def analyze(self, text):
        raise AssertionError("detector.analyze must not be called when the flag is off")


class _StubDetector:
    def __init__(self, result):
        self.result = result

    async def analyze(self, text):
        return self.result


def _make_service(enabled=False, detector=None):
    if detector is None and enabled:
        detector = LanguageDetector(llm=None)
    service = NotebookLessonService(
        _FakeLLM(), FakeGuard(), _FakeRetriever(),
        language_detector=detector,
        language_detection_enabled=enabled,
    )
    return service


@pytest.mark.asyncio
async def test_detection_short_circuits_before_analyze_when_flag_off():
    service = _make_service(enabled=False, detector=_BombDetector())
    assert await service._detected_language("کیا یہ سبق ہے؟") is None


@pytest.mark.asyncio
async def test_detection_returns_none_for_low_confidence():
    service = _make_service(enabled=True, detector=_StubDetector(("English", 0.0)))
    assert await service._detected_language("not confident") is None


@pytest.mark.asyncio
async def test_detection_returns_confident_language():
    service = _make_service(enabled=True, detector=_StubDetector(("Roman Urdu", 0.85)))
    assert await service._detected_language("kya baat hai") == "Roman Urdu"


def test_response_language_is_none_when_flag_off():
    service = _make_service(enabled=False)
    lesson = NotebookLesson(session_id="s", course_id="c", topic="t", language="English", lecture_text="", chunks=[], sources=[])
    assert service.response_language(lesson) is None


def test_response_language_advertises_active_language_when_flag_on():
    service = _make_service(enabled=True)
    lesson = NotebookLesson(session_id="s", course_id="c", topic="t", language="Roman Urdu", lecture_text="", chunks=[], sources=[])
    assert service.response_language(lesson) == "Roman Urdu"


@pytest.mark.asyncio
async def test_start_fast_path_keeps_requested_language_when_flag_off():
    service = _make_service(enabled=False, detector=_BombDetector())
    response = await service.start("course", "Table of 9", "English")
    assert response.response_type == "multiplication_table"
    assert response.language is None
    assert service.store.get(response.session_id).language == "English"


@pytest.mark.asyncio
async def test_start_fast_path_advertises_language_when_flag_on_english():
    service = _make_service(enabled=True)
    response = await service.start("course", "Table of 9", "English")
    assert response.language == "English"
    assert service.store.get(response.session_id).language == "English"


@pytest.mark.asyncio
async def test_interrupt_normal_branch_answers_roman_urdu_but_keeps_lesson_language():
    service = _make_service(enabled=True)
    lesson = NotebookLesson(
        session_id="s", course_id="c", topic="Algebra",
        language="English", lecture_text="Current lecture text.",
        chunks=[WhiteboardBlock(id="b1", type="text", text="Line one.", status="complete")],
        sources=[], history=[],
    )
    service.store.put(lesson)
    response = await service.interrupt("s", "mujhe samajh nahi aya, aap kya samjhate hain", 0)
    assert service.store.get("s").language == "English"
    assert response.language == "English"
    assert response.answer_language == "Roman Urdu"
    assert response.answer.text == "Spoken answer text"


@pytest.mark.asyncio
async def test_interrupt_english_turn_does_not_switch_language():
    service = _make_service(enabled=True)
    lesson = NotebookLesson(
        session_id="s", course_id="c", topic="Algebra",
        language="English", lecture_text="Current lecture text.",
        chunks=[WhiteboardBlock(id="b1", type="text", text="Line one.", status="complete")],
        sources=[], history=[],
    )
    service.store.put(lesson)
    question = "can you please explain this step again"
    assert any(word in ENGLISH_STOPWORDS for word in question.split())
    response = await service.interrupt("s", question, 0)
    assert service.store.get("s").language == "English"
    assert response.language == "English"


@pytest.mark.asyncio
async def test_interrupt_checkin_reply_answers_roman_urdu_but_keeps_lesson_language():
    service = _make_service(enabled=True)
    lesson = NotebookLesson(
        session_id="s", course_id="c", topic="Algebra",
        language="English", lecture_text="",
        chunks=[], sources=[], history=[],
        awaiting_checkin_reply=True,
        last_checkin_question="Are you following along ok?",
        last_checkin_directive="The student looks confused.",
    )
    service.store.put(lesson)
    response = await service.interrupt("s", "mujhe yeh theek se nahi samajh aya", 0)
    assert service.store.get("s").language == "English"
    assert service.store.get("s").awaiting_checkin_reply is False
    assert response.interrupt_type == "checkin"
    assert response.language == "English"
    assert response.answer_language == "Roman Urdu"
    assert response.answer.text == "Spoken answer text"


@pytest.mark.asyncio
async def test_interrupt_does_not_switch_when_flag_off():
    service = _make_service(enabled=False, detector=_BombDetector())
    lesson = NotebookLesson(
        session_id="s", course_id="c", topic="Algebra",
        language="English", lecture_text="Current lecture text.",
        chunks=[WhiteboardBlock(id="b1", type="text", text="Line one.", status="complete")],
        sources=[], history=[],
    )
    service.store.put(lesson)
    response = await service.interrupt("s", "mujhe samajh nahi aya, aap kya samjhate hain", 0)
    assert service.store.get("s").language == "English"
    assert response.language is None