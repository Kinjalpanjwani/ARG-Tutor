"""Language-aware output routing: voice selection and prompt directives."""


def voice_for(language: str, *, english_voice: str, urdu_voice: str) -> str:
    """Pick the TTS voice for an active language.

    Roman Urdu collapses to the Urdu voice. Unknown or English input uses the
    English voice, preserving current behavior.
    """
    if language in ("Urdu", "Roman Urdu"):
        return urdu_voice
    return english_voice


def output_language(language: str) -> str:
    """Collapse Roman Urdu to Urdu for LLM output and voice routing."""
    if language == "Roman Urdu":
        return "Urdu"
    return language


def roman_urdu_instruction(language: str) -> str:
    """Output-language directive for whiteboard/answer text rendered to the student.

    The student reads the whiteboard, so urdu-family replies are written in
    Roman Urdu (Latin letters) rather than Urdu script, matching the per-user
    override while the urdu TTS voice keeps speaking the same language.
    """
    if language in ("Urdu", "Roman Urdu"):
        return (
            "Respond entirely in Roman Urdu -- Urdu words written in Latin letters "
            "(like 'aap', 'samajh', 'dubara'), not in Urdu script. Keep technical "
            "terms, equations, names, and key English vocabulary in English where "
            "that is the natural terminology."
        )
    return (
        "Always respond only in English, regardless of the language or script the "
        "student types or speaks. Do not mirror their language and do not switch to "
        "any other language, even if the student writes in Urdu or Roman Urdu."
    )


def language_instruction(language: str) -> str:
    """Output-language directive to append to a prompt.

    Roman Urdu collapses to Urdu; only English and Urdu directives are produced.
    """
    if language in ("Urdu", "Roman Urdu"):
        return (
            "Respond entirely in Urdu, written in Urdu script. Keep technical terms, "
            "equations, names, and key English vocabulary in English where that is the "
            "natural terminology; otherwise use correct Urdu terminology, not literal "
            "word-for-word translation. Even if the student writes in Roman Urdu, "
            "respond in Urdu script."
        )
    return (
        "Always respond only in English, regardless of the language or script the "
        "student types or speaks. Do not mirror their language and do not switch to "
        "any other language, even if the student writes in Urdu or Roman Urdu."
    )