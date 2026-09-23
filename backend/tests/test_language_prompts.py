import pytest

from app.language.routing import language_instruction, output_language, voice_for
from app.teaching.notebook_lesson import (
    INTERRUPTION_PROMPT,
    INTERRUPTION_PROMPT_URDU,
    LECTURE_PERSONA,
    LECTURE_PERSONA_URDU,
    _interruption_prompt_for,
    _lecture_persona_for,
)
from app.teaching.vision import build_checkin_followup_prompt, build_checkin_prompt


def test_output_language_collapses_roman_urdu():
    assert output_language("Roman Urdu") == "Urdu"
    assert output_language("Urdu") == "Urdu"
    assert output_language("English") == "English"


def test_language_instruction_urdu_requests_urdu_script():
    instruction = language_instruction("Urdu")
    assert "Urdu script" in instruction
    assert "English" in instruction or "English" in instruction


def test_language_instruction_roman_urdu_collapses_to_urdu():
    assert language_instruction("Roman Urdu") == language_instruction("Urdu")


def test_language_instruction_english_reproduces_english_rule():
    instruction = language_instruction("English")
    assert "respond only in English" in instruction


def test_voice_for_urdu_and_roman_urdu_use_urdu_voice():
    english = "en-US-AriaNeural"
    urdu = "ur-PK-AsadNeural"
    assert voice_for("Urdu", english_voice=english, urdu_voice=urdu) == urdu
    assert voice_for("Roman Urdu", english_voice=english, urdu_voice=urdu) == urdu
    assert voice_for("English", english_voice=english, urdu_voice=urdu) == english


def test_lecture_persona_english_is_unchanged_constant():
    assert _lecture_persona_for("English") == LECTURE_PERSONA
    assert "never mirror their language" in LECTURE_PERSONA


def test_lecture_persona_roman_urdu_collapses_to_urdu_variant():
    assert _lecture_persona_for("Roman Urdu") == LECTURE_PERSONA_URDU
    assert _lecture_persona_for("Urdu") == LECTURE_PERSONA_URDU
    assert "Urdu script" in LECTURE_PERSONA_URDU


def test_lecture_persona_english_has_no_urdu_instruction():
    assert "Urdu script" not in LECTURE_PERSONA


def test_interruption_prompt_selection():
    assert _interruption_prompt_for("English") == INTERRUPTION_PROMPT
    assert _interruption_prompt_for("Urdu") == INTERRUPTION_PROMPT_URDU
    assert "Urdu script" in INTERRUPTION_PROMPT_URDU


def test_build_checkin_prompt_default_is_english():
    prompt = build_checkin_prompt("The student looks confused.")
    assert "check-in question" in prompt
    assert "Urdu" not in prompt


def test_build_checkin_prompt_urdu_adds_language_note():
    prompt = build_checkin_prompt("The student looks confused.", language="Roman Urdu")
    assert "Urdu script" in prompt


def test_build_checkin_followup_english_default_is_unchanged():
    prompt = build_checkin_followup_prompt(
        "Are you following along ok?",
        "The student looks confused.",
        "yes",
        "Here is the point.",
    )
    assert "Always reply in English" in prompt
    assert "Urdu" not in prompt


def test_build_checkin_followup_urdu_replaces_english_rule():
    prompt = build_checkin_followup_prompt(
        "Are you following along ok?",
        "The student looks confused.",
        "yes",
        "Here is the point.",
        language="Urdu",
    )
    assert "Always reply in English" not in prompt
    assert "Urdu script" in prompt


def test_build_checkin_followup_no_reply_branch_urdu():
    prompt = build_checkin_followup_prompt(
        "Are you following along ok?",
        "The student looks confused.",
        None,
        "Here is the point.",
        language="Roman Urdu",
    )
    assert "Always reply in English" not in prompt
    assert "Urdu script" in prompt