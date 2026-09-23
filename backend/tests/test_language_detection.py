import pytest

from app.llm.groq_client import GroqServiceError
from app.language.detector import (
    SWITCH_CONFIDENCE_MIN,
    LanguageDetector,
    lexicon_language,
    script_language,
)


class _FakeGroq:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def json(self, *args, **kwargs):
        self.calls += 1
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def test_script_language_detects_urdu_letters_only():
    assert script_language("کیا آپ مجھے سمجھا سکتے ہیں؟") == "Urdu"
    assert script_language("میرا نام احمد ہے۔") == "Urdu"
    assert script_language("plain English text") is None
    assert script_language("") is None


@pytest.mark.asyncio
async def test_urdu_script_detects_without_any_llm_call():
    detector = LanguageDetector(llm=None)
    language, confidence = await detector.analyze("کیا آپ مجھے سمجھا سکتے ہیں؟")
    assert language == "Urdu"
    assert confidence >= SWITCH_CONFIDENCE_MIN


@pytest.mark.asyncio
async def test_clear_english_detects_without_llm_call():
    detector = LanguageDetector(llm=None)
    language, confidence = await detector.analyze(
        "Can you please explain photosynthesis using a simple example?"
    )
    assert language == "English"
    assert confidence >= SWITCH_CONFIDENCE_MIN


@pytest.mark.asyncio
async def test_roman_urdu_detects_via_lexicon_without_llm_call():
    detector = LanguageDetector(llm=None)
    language, confidence = await detector.analyze(
        "mujhe yeh samajh nahi aa raha hai, aap kya samjhate hain"
    )
    assert language == "Roman Urdu"
    assert confidence >= SWITCH_CONFIDENCE_MIN


@pytest.mark.asyncio
async def test_ambiguous_latin_sentence_uses_classifier_for_roman_urdu():
    fake = _FakeGroq({"language": "Roman Urdu"})
    detector = LanguageDetector(llm=fake)
    language, confidence = await detector.analyze("yaar honestly baat hai that simple")
    assert fake.calls == 1
    assert language == "Roman Urdu"
    assert confidence >= SWITCH_CONFIDENCE_MIN


@pytest.mark.asyncio
async def test_ambiguous_latin_sentence_uses_classifier_for_english():
    fake = _FakeGroq({"language": "English"})
    detector = LanguageDetector(llm=fake)
    language, _ = await detector.analyze("the quick brown basically fox hmm thing")
    assert fake.calls == 1
    assert language == "English"


@pytest.mark.asyncio
async def test_classifier_failure_falls_back_to_english_low_confidence():
    detector = LanguageDetector(llm=_FakeGroq(GroqServiceError("boom")))
    language, confidence = await detector.analyze("something weird yaar hmm what")
    assert language == "English"
    assert confidence < SWITCH_CONFIDENCE_MIN


@pytest.mark.asyncio
async def test_empty_and_numeric_inputs_return_english_no_signal():
    detector = LanguageDetector(llm=None)
    assert await detector.analyze("") == ("English", 0.0)
    assert await detector.analyze("   ") == ("English", 0.0)
    language, confidence = await detector.analyze("1234 5678 12.5")
    assert language == "English"
    assert confidence == 0.0


def test_lexicon_prefilter_ratios():
    language, confidence = lexicon_language(["mujhe", "yeh", "samajh", "nahi", "aata"])
    assert language == "Roman Urdu"
    assert confidence >= SWITCH_CONFIDENCE_MIN

    language, _ = lexicon_language(["what", "is", "the", "meaning", "of", "this", "word"])
    assert language == "English"

    language, confidence = lexicon_language([])
    assert language is None
    assert confidence == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("phrase", [
    "2 ka table batao",
    "sir loi samajh nahi aya",
    "kya aap isay zyada asaan alfaaz mein samjha sakte hain",
    "wapis samjha do",
])
async def test_short_roman_urdu_phrases_hit_lexicon_without_classifier(phrase):
    detector = LanguageDetector(llm=_BombGroq())
    language, confidence = await detector.analyze(phrase)
    assert language == "Roman Urdu"
    assert confidence >= SWITCH_CONFIDENCE_MIN


class _BombGroq:
    async def json(self, *args, **kwargs):
        raise AssertionError("short-RU phrase must be resolved by the lexicon")