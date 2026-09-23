"""Cheap-first, per-turn language detection: Urdu script regex -> Roman-Urdu
lexicon pre-filter -> Groq classifier. Returns (language, confidence) where
language is one of English, Urdu, Roman Urdu.
"""

import logging
import re

from app.language.roman_urdu_lexicon import (
    EN_CONFIDENT_RATIO,
    ENGLISH_STOPWORDS,
    RU_CONFIDENT_RATIO,
    ROMAN_URDU_LEXICON,
)
from app.llm.groq_client import GroqService, GroqServiceError

logger = logging.getLogger(__name__)

LANGUAGES = ("English", "Urdu", "Roman Urdu")

URDU_SCRIPT_REGEX = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")

_LATIN_WORD_REGEX = re.compile(r"[A-Za-z]+")

SWITCH_CONFIDENCE_MIN = 0.6

_CLASSIFIER_PROMPT = """
Classify the dominant language of the learner's message below. The message was
typed by a Pakistani learner, so it is one of:
- "English": plain English text.
- "Roman Urdu": Urdu spoken or typed using English/Latin letters, often mixed
  with some English words. Examples: "kya aap mujhe samjha sakte hain",
  "yeh theek hai", "mujhe iska matlab batao".
Return JSON only: {"language": "English"|"Roman Urdu"}.
"""


def script_language(text: str) -> str | None:
    """Return 'Urdu' when the text contains Urdu/Arabic-script characters."""
    if not text:
        return None
    return "Urdu" if URDU_SCRIPT_REGEX.search(text) else None


def lexicon_language(latin_words: list[str]) -> tuple[str | None, float]:
    """Pre-filter a Latin-script turn using keyword ratios.

    Returns (language, confidence) with language None when inconclusive, in
    which case the caller may fall through to the classifier.
    """
    total = len(latin_words)
    if total == 0:
        return None, 0.0
    ru_hits = sum(1 for word in latin_words if word in ROMAN_URDU_LEXICON)
    en_hits = sum(1 for word in latin_words if word in ENGLISH_STOPWORDS)
    ru_ratio = ru_hits / total
    if ru_ratio >= RU_CONFIDENT_RATIO:
        return "Roman Urdu", 0.85
    if ru_hits == 0 and en_hits / total >= EN_CONFIDENT_RATIO:
        return "English", 0.9
    return None, 0.0


class LanguageDetector:
    """Cost-cheapest-first cascade; the Groq classifier is the last resort, used
    only for ambiguous Latin-script turns."""

    def __init__(self, llm: GroqService | None = None) -> None:
        self.llm = llm

    async def analyze(self, text: str) -> tuple[str, float]:
        if not text or not text.strip():
            return "English", 0.0

        script_lang = script_language(text)
        if script_lang is not None:
            return "Urdu", 0.99

        latin_words = [word.lower() for word in _LATIN_WORD_REGEX.findall(text)]
        if not latin_words:
            return "English", 0.0

        language, confidence = lexicon_language(latin_words)
        if language is not None:
            return language, confidence

        if self.llm is None:
            return "English", 0.0
        return await self._classify(text)

    async def _classify(self, text: str) -> tuple[str, float]:
        try:
            payload = await self.llm.json([
                {"role": "system", "content": _CLASSIFIER_PROMPT},
                {"role": "user", "content": f"Message: {text}"},
            ], temperature=0.0)
        except GroqServiceError as exc:
            logger.warning("[LANG] classifier failed; falling back to English: %s", exc)
            return "English", 0.0
        language = (payload or {}).get("language") if isinstance(payload, dict) else None
        if language == "Urdu":
            return "Urdu", 0.9
        if language == "Roman Urdu":
            return "Roman Urdu", 0.75
        if language == "English":
            return "English", 0.75
        return "English", 0.0