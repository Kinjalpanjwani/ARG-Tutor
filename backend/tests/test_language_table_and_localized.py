import re

import pytest

from app.language.detector import LanguageDetector
from app.teaching.notebook_lesson import NotebookLessonService, localized


class _FakeGuard:
    async def classify(self, topic):
        return type("Decision", (), {"allowed": True})()


class _FakeLLM:
    async def json(self, *args, **kwargs):
        raise AssertionError("fast-path table must not call the LLM")

    async def text(self, *args, **kwargs):
        raise AssertionError("fast-path table must not call the LLM")


def _service(enabled=True):
    retriever = type("R", (), {"store": type("S", (), {"records": []})()})()
    return NotebookLessonService(
        _FakeLLM(), _FakeGuard(), retriever,
        language_detector=LanguageDetector(llm=None),
        language_detection_enabled=enabled,
    )


def test_table_fast_path_english_is_byte_identical():
    kind, blocks, follow_up = NotebookLessonService._table_fast_path("table of 2?", "English")
    assert kind == "multiplication_table"
    assert blocks[0].type == "heading"
    assert blocks[0].text == "Table of 2"
    assert [block.text for block in blocks[1:6]] == [f"2 × {i} = {2 * i}" for i in range(1, 6)]
    assert blocks[6].type == "checkin"
    assert blocks[6].text == "Now you tell me: what is 7 × 2?"
    assert [block.text for block in blocks[7:12]] == [f"2 × {i} = {2 * i}" for i in range(6, 11)]
    assert blocks[12].type == "note"
    assert blocks[12].text == "Want to practice it?"
    assert follow_up == "Can you continue the table of 2 beyond 10?"


@pytest.mark.parametrize("topic", [
    "mujhe 2 ka table batao",
    "2 ka table batao",
    "5 ka pahada",
    "7 ka pahra",
    "8 ka pahad",
    "9 ka pahar",
    "6 ka phara",
])
def test_table_fast_path_matches_roman_urdu_phrasings(topic):
    kind, blocks, follow_up = NotebookLessonService._table_fast_path(topic, "Roman Urdu")
    number = int(re.search(r"\d+", topic).group())
    assert kind == "multiplication_table"
    assert len(blocks) == 13
    assert blocks[0].text == f"{number} ka pahada"
    assert [block.text for block in blocks[1:6]] == [f"{number} × {i} = {number * i}" for i in range(1, 6)]
    assert blocks[6].type == "checkin"
    assert blocks[6].text == f"Ab aap batayein: 7 × {number} kya hai?"
    assert [block.text for block in blocks[7:12]] == [f"{number} × {i} = {number * i}" for i in range(6, 11)]
    assert blocks[-1].text == "Kya aap is ki mashq karna chahenge?"
    assert follow_up == f"Kya aap {number} ka pahada 10 say aage jari rakh saktay hain?"


def test_table_fast_path_urdu_uses_script():
    kind, blocks, follow_up = NotebookLessonService._table_fast_path("5 ka pahada", "Urdu")
    assert kind == "multiplication_table"
    assert "پہاڑا" in blocks[0].text
    assert "مشق" in blocks[-1].text
    assert "پہاڑا" in follow_up


@pytest.mark.asyncio
async def test_start_roman_urdu_table_is_deterministic_and_no_llm():
    service = _service()
    response = await service.start("course", "mujhe 2 ka table batao", "English")
    assert response.response_type == "multiplication_table"
    assert response.language == "Roman Urdu"
    assert service.store.get(response.session_id).language == "Roman Urdu"
    assert [block.text for block in response.chunks if block.type == "table"] == [f"2 × {i} = {2 * i}" for i in range(1, 11)]
    assert response.chunks[0].text == "2 ka pahada"
    assert [block.type for block in response.chunks].count("checkin") == 1
    assert response.follow_up_questions and "pahada" in response.follow_up_questions[0]


@pytest.mark.asyncio
async def test_start_english_table_keeps_english_when_flag_off():
    service = _service(enabled=False)
    response = await service.start("course", "table of 3", "English")
    assert response.response_type == "multiplication_table"
    assert response.language is None
    assert response.chunks[0].text == "Table of 3"
    assert response.chunks[-1].text == "Want to practice it?"


EN_COPY = "english-context"
UR_COPY = "urdu-context"
COPY = [
    ("Still there? Pausing until you're back.", "کیا آپ وہاں ہیں؟ آپ کے واپس آنے تک میں رکی رہوں گا۔"),
    ("Welcome back — let's pick up where we left off.", "خوش آمدید — چلیں وہیں سے جاری رکھتے ہیں جہاں رکے تھے۔"),
    ("Are you following along okay, or would you prefer a simpler example?", "کیا آپ سمجھ رہے ہیں؟ یا کوئی آسان مثال چاہیں گے؟"),
    ("Continuing the lesson…", "سبق جاری رکھ رہے ہیں…"),
]


@pytest.mark.parametrize("english,urdu", COPY)
def test_localized_returns_english_verbatim_for_english(english, urdu):
    assert localized("English", english, urdu) == english
    assert localized("english", english, urdu) == english


@pytest.mark.parametrize("english,urdu", COPY)
def test_localized_returns_urdu_for_urdu_and_roman_urdu(english, urdu):
    assert localized("Urdu", english, urdu) == urdu
    assert localized("Roman Urdu", english, urdu) == urdu